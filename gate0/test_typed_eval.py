#!/usr/bin/env python3
"""Small deterministic tests for Gate 0 typed transition evaluation."""

from typed_eval import evaluate_transition, state_from_question


def assert_failure(name, expected, question, response, action_name, history_queries=None, open_gaps=None):
    result = evaluate_transition(
        question=question,
        state=state_from_question(question, open_gaps=open_gaps),
        action_name=action_name,
        response=response,
        history_queries=history_queries or [],
    )
    actual = result.failure_type
    if actual != expected:
        raise AssertionError(
            f"{name}: expected {expected}, got {actual}; "
            f"phi_q={result.phi_q}, phi_c={result.phi_c}, phi_s={result.phi_s}, "
            f"details={result.details}"
        )


def main():
    question = "Shirley Temple was appointed Chief of Protocol in 1976 by which president?"

    assert_failure(
        name="good_query",
        expected="success",
        question=question,
        action_name="reasoning",
        response="So the next query is <query>Shirley Temple Chief of Protocol 1976 president</query>",
    )

    assert_failure(
        name="query_missing_bridge_entity",
        expected="query_fail",
        question=question,
        action_name="reasoning",
        response="So the next query is <query>United States president list</query>",
    )

    assert_failure(
        name="repeated_query",
        expected="query_fail",
        question=question,
        action_name="reasoning",
        history_queries=["Shirley Temple Chief of Protocol 1976 president"],
        response="So the next query is <query>Shirley Temple Chief of Protocol 1976 president</query>",
    )

    assert_failure(
        name="none_evidence",
        expected="query_fail",
        question=question,
        action_name="document_analysis",
        response="Based on the query, the relevant evidence is <evidence>None</evidence>.",
    )

    assert_failure(
        name="unsupported_nonempty_evidence",
        expected="success",
        question=question,
        action_name="document_analysis",
        response="Based on the query, the relevant evidence is <evidence>Shirley Temple was an actress.</evidence>.",
    )

    assert_failure(
        name="premature_stop",
        expected="stop_fail",
        question=question,
        action_name="reasoning",
        open_gaps=["Which president appointed Shirley Temple?"],
        response="So the answer is <answer>Gerald Ford</answer>",
    )

    assert_failure(
        name="supported_evidence",
        expected="success",
        question=question,
        action_name="document_analysis",
        response=(
            "Based on the query, the relevant evidence is "
            "<evidence>Shirley Temple was appointed Chief of Protocol in 1976 by President Gerald Ford.</evidence>."
        ),
    )

    assert_failure(
        name="mixed_query_and_stop",
        expected="mixed",
        question=question,
        action_name="reasoning",
        open_gaps=["Which president appointed Shirley Temple?"],
        response=(
            "So the next query is <query>United States president list</query> "
            "So the answer is <answer>Gerald Ford</answer>"
        ),
    )

    print("All typed_eval tests passed.")


if __name__ == "__main__":
    main()
