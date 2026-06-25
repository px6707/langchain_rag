from app.schemas.retrieval import RetrievalPlan
from app.services.retrieval_rules import rule_postcheck_retrieve, rule_precheck


def test_rule_precheck_greeting():
    plan = rule_precheck("你好")
    assert plan is not None
    assert plan.action == "skip"


def test_rule_postcheck_forces_retrieve_on_doc_intent():
    skip_plan = RetrievalPlan(action="skip", reason="llm skip")
    plan = rule_postcheck_retrieve("请根据上传文档说明合同条款", skip_plan)
    assert plan.action == "retrieve"


def test_rule_postcheck_forces_retrieve_on_anaphora():
    skip_plan = RetrievalPlan(action="skip", reason="llm skip")
    plan = rule_postcheck_retrieve("那它的限制是什么？", skip_plan)
    assert plan.action == "retrieve"


def test_rule_postcheck_keeps_skip_for_chitchat():
    skip_plan = RetrievalPlan(action="skip", reason="llm skip")
    plan = rule_postcheck_retrieve("哈哈", skip_plan)
    assert plan.action == "skip"
