"""Readable answers assembled from recomputed results, including partial-data cases."""

from typing import Any

from ai_agent_template.commerce.operations_models import AnswerParagraph, InvestigationRequest


def number(value: float | int) -> str:
    return f"{value:,.2f}".rstrip("0").rstrip(".") if isinstance(value, float) else f"{value:,}"


def percent(value: float) -> str:
    return number(round(value * 100, 2)) + "%"


def grounded_answer(
    request: InvestigationRequest, results: dict[str, Any], failures: dict[str, str]
) -> list[AnswerParagraph]:
    paragraphs: list[AnswerParagraph] = []
    advice: list[str] = []
    limits: list[str] = []
    listing = results.get("listing")
    if listing is not None:
        issues = listing.get("issues", [])
        if issues:
            details = []
            for item in issues[:3]:
                field = {
                    "capacity": "容量",
                    "package_quantity": "包装数量",
                    "product_version": "商品版本",
                    "dimensions": "尺寸",
                }.get(item["field"], item["field"])
                if item["issue_type"] == "imported_issue":
                    details.append(f"导入的问题报告提示{field}：{item['reason']}（尚未独立确认）")
                elif item["actual_value"] is None:
                    details.append(
                        f"Listing 缺少{field}"
                        + (
                            f"，来源规格为 {item['expected_value']}"
                            if item["expected_value"] is not None
                            else ""
                        )
                    )
                else:
                    details.append(
                        f"{field}在 Listing 中写为 {item['actual_value']}，来源规格为 {item['expected_value']}"
                    )
            paragraphs.append(
                AnswerParagraph(
                    text="商品信息存在需要复核的差异：" + "；".join(details) + "。",
                    evidence_ids=[item["source_reference"] for item in issues[:3]],
                )
            )
            advice.append(
                "先核对供应商权威规格和当前销售版本，再人工确认是否修正 Listing 中的差异字段"
            )
        elif listing.get("missing_evidence"):
            limits.append("Listing 资料不完整，暂时无法判断商品描述是否与实际规格一致")
        else:
            paragraphs.append(
                AnswerParagraph(
                    text="在当前有效的供应商规格与 Listing 快照之间，没有发现字段差异。这是内部资料一致性检查，不能据此判断平台合规。"
                )
            )

    returns = results.get("returns")
    if returns is not None:
        refs = [returns["evidence"][0]["evidence_id"]]
        units, sold, rate = returns["returned_units"], returns["sold_units"], returns["return_rate"]
        text = (
            f"退货报表记录了 {units} 件退货"
            if units
            else "当前资料未观测到退货件数，不能据此断定没有退货"
        )
        text += f"，销售窗口内报表记录 {sold} 件销售" if sold is not None else "，销售分母缺失"
        text += (
            f"，按所选口径计算的退货率为 {percent(rate)}"
            if rate is not None
            else "，目前无法计算有效退货率，不能按零退货率处理"
        )
        before = returns.get("comparison")
        if before and before["return_rate"] is not None and rate is not None:
            old = before["return_rate"]
            direction = "高于" if rate > old else "低于" if rate < old else "与"
            text += f"；对照期为 {percent(old)}，本期{direction}对照期" + (
                "持平" if rate == old else ""
            )
            refs.append(before["evidence"][0]["evidence_id"])
        reasons = returns.get("reasons", {})
        if reasons:
            reason, count = max(reasons.items(), key=lambda item: item[1])
            label = {
                "package_damaged": "包装损坏",
                "damaged": "损坏",
                "changed_mind": "买家改变主意",
                "defective": "商品缺陷",
                "not_as_described": "与描述不符",
            }.get(reason, reason)
            text += f"。记录中件数最多的退货原因为“{label}”（{count} 件）"
        if returns.get("timeline"):
            text += "。同一窗口还存在商品或包装变更记录，但时间上的共现不足以证明变更导致退货"
            refs.append(returns["evidence"][-1]["evidence_id"])
        paragraphs.append(AnswerParagraph(text=text + "。", evidence_ids=refs))
        if request.return_basis == "event_period":
            limits.append(
                "这里的退货率按事件日期统计，退货可能来自更早的订单，不能直接当作同批订单的退货概率"
            )
        if units:
            advice.append(
                "把退货订单对应到销售批次、包装版本和物流/质检记录，优先核实件数最多的退货原因"
            )

    ads = results.get("ads")
    if ads is not None:
        groups = ads.get("groups", [])
        if not groups:
            limits.append("所选窗口没有广告报告，暂时无法判断广告表现")
        for group in groups[:3]:
            refs = [group["evidence_id"]]
            currency, totals, metrics = group["currency"], group["totals"], group["metrics"]
            identity = f"广告活动 {group['campaign_id']} / 广告组 {group['ad_group_id']}"
            if group.get("search_term"):
                identity += f" / 搜索词“{group['search_term']}”"
            text = f"{identity}（{currency}、{group['ad_type']}、{group['attribution_window']} 天归因）"
            facts = []
            for field, label in (("spend", "花费"), ("attributed_sales", "归因销售额")):
                if totals[field] is not None:
                    facts.append(f"{label} {number(totals[field])} {currency}")
            facts.append(
                f"ACOS 为 {percent(metrics['ACOS'])}"
                if metrics["ACOS"] is not None
                else "ACOS 缺少有效分母，无法计算"
            )
            if metrics["CVR"] is not None:
                facts.append(f"转化率为 {percent(metrics['CVR'])}")
            text += "，" + "，".join(facts)
            keys = (
                "currency",
                "ad_type",
                "attribution_window",
                "campaign_id",
                "ad_group_id",
                "search_term",
            )
            before = next(
                (
                    item
                    for item in ads.get("comparison_groups", [])
                    if all(item[key] == group[key] for key in keys)
                ),
                None,
            )
            if before and before["metrics"]["ACOS"] is not None and metrics["ACOS"] is not None:
                text += f"；同口径对照期 ACOS 为 {percent(before['metrics']['ACOS'])}"
                refs.append(before["evidence_id"])
            if group["attribution_immature"] or group["insufficient_sample"]:
                text += "。这组数据存在归因未成熟或样本不足，需要等待归因完成并复核样本"
            paragraphs.append(AnswerParagraph(text=text + "。", evidence_ids=refs))
        if len(groups) > 3:
            limits.append("以上说明前三组广告数据，其余分组见下方明细；不同币种和归因口径分别判断")
        if ads.get("anomalies"):
            advice.append(
                "复核异常广告组的高消耗搜索词、商品匹配和归因成熟度，再决定是否调整竞价或否词"
            )
        terms = [
            item
            for item in ads.get("search_term_metrics", [])
            if item.get("search_term") and item["totals"]["spend"] is not None
        ]
        if terms:
            term = terms[0]
            paragraphs.append(
                AnswerParagraph(
                    text=f"搜索词“{term['search_term']}”在广告活动 {term['campaign_id']} / 广告组 {term['ad_group_id']} 的报表中花费 {number(term['totals']['spend'])} {term['currency']}（{term['ad_type']}、{term['attribution_window']} 天归因）。这可以作为复核的线索，尚不能单凭消耗判断该词与商品是否匹配。",
                    evidence_ids=[term["evidence_id"]],
                )
            )

    knowledge = results.get("knowledge")
    if knowledge is not None:
        if not knowledge.get("sources"):
            limits.append("尚未检索到适用于当前版本和日期的文档，因此无法用文档进一步解释原因")
        elif not paragraphs:
            for source in knowledge["sources"][:2]:
                paragraphs.append(
                    AnswerParagraph(
                        text=f"当前检索到的相关资料“{source['source']}”记载：\n{source['content'][:400]}",
                        evidence_ids=[source["evidence"]["evidence_id"]],
                    )
                )
            limits.append(
                "以上是相关原文片段；未启用或未通过模型核验时，不进一步推断资料未写明的原因"
            )
    if failures:
        names = {
            "listing": "Listing 检查",
            "returns": "退货分析",
            "ads": "广告分析",
            "knowledge": "资料检索",
        }
        failed = [names[key] for key in failures if key in names]
        if failed:
            limits.append("、".join(failed) + "执行失败，这些部分需要重试后再判断")
    if not paragraphs:
        limits.insert(
            0, "目前的资料还不足以直接回答这个问题，需要补充与所选商品、日期相关的报表或有效文档"
        )
    if limits:
        paragraphs.append(
            AnswerParagraph(kind="limitation", text="判断的边界是：" + "；".join(limits) + "。")
        )
    if advice:
        paragraphs.append(
            AnswerParagraph(
                kind="suggestion", text="建议按这个顺序处理：" + "；然后".join(advice) + "。"
            )
        )
    elif paragraphs and any(item.kind == "fact" for item in paragraphs):
        paragraphs.append(
            AnswerParagraph(
                kind="suggestion",
                text="建议先复核对应报表和资料版本；当前检查没有提供需要立即修改资料或投放的充分依据。",
            )
        )
    return paragraphs
