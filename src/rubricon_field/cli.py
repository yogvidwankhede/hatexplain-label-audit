"""Command line entry point: ``rubricon-field run`` and ``rubricon-field report``.

Results are written through ``rubricon.core.store.Store``, which records a
manifest with a content hash for every file. That is not decoration: it is what
lets the README assert that a number came from a particular set of bytes, and it
is what makes "the numbers changed and nobody noticed" a detectable event.

Determinism
-----------
There is no timestamp anywhere in the output and no unseeded randomness in the
pipeline. Running ``rubricon-field run`` twice on the same input produces
byte-identical files and therefore identical manifest hashes. That is checked in
the test suite rather than asserted here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from rubricon.core.store import Store

from . import __version__
from . import assumptions as A
from .data import NO_MAJORITY, SPLIT_2_1, UNANIMOUS, load_hatexplain
from .study_a import run_study_a
from .study_b import run_study_b
from .study_c import run_study_c

DEFAULT_DATA = "data/hatexplain.json"
DEFAULT_RESULTS = "results"


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------


def build_summary(a: dict, b: dict, c: dict) -> dict:
    """The flat block of headline numbers every prose claim in the README maps to.

    Its existence is a discipline, not a convenience: if a sentence in the README
    quotes a number, that number is here, and if it is not here it was not
    computed and must not be written.
    """
    battery = a["overall_agreement"]["battery"]
    cons = a["consensus_structure"]["three_way"]
    contrast = a["binary_contrast"]["contrast"]
    pool = a["annotators"]["pool"]
    loo = a["annotators"]["leave_one_annotator_out"]["rows"][0]
    ceiling = a["accuracy_ceiling"]
    mde = ceiling["minimum_detectable_effect"]
    b3 = b["three_way"]
    proj = c["reliability_projection"]
    alloc = c["budget_allocation"]
    costs = c["cost_model"]

    return {
        "dataset": a["dataset"],
        "assumptions": A.assumptions_block(),
        "study_a": {
            "krippendorff_alpha_nominal": battery["krippendorff_alpha"]["value"],
            "alpha_ci": battery["ci"],
            "percent_agreement": battery["percent_agreement"]["value"],
            "fleiss_kappa": battery["fleiss_kappa"]["value"],
            "gwet_ac1": battery["gwet_ac1"]["value"],
            "kappa_paradox_detected": a["overall_agreement"]["kappa_paradox"]["detected"],
            "unanimous_posts": cons["counts"][UNANIMOUS],
            "unanimous_share": cons["percentages"][UNANIMOUS],
            "split_2_1_posts": cons["counts"][SPLIT_2_1],
            "split_2_1_share": cons["percentages"][SPLIT_2_1],
            "no_majority_posts": cons["counts"][NO_MAJORITY],
            "no_majority_share": cons["percentages"][NO_MAJORITY],
            "one_vote_fragile_share": cons["one_vote_fragile_fraction"],
            "alpha_binary": contrast["alpha_binary"],
            "alpha_delta_binary_minus_three_way": contrast["alpha_delta"],
            "raw_agreement_binary": contrast["raw_binary"],
            "alpha_offensive_vs_hatespeech_conditional": contrast[
                "alpha_offensive_vs_hatespeech_conditional"
            ],
            "busiest_annotator": pool["busiest_annotator"],
            "top5_share": pool["top5_share"],
            "leave_out_busiest_alpha_delta": loo["alpha_delta"],
            "leave_out_busiest_gold_labels_lost": loo["gold_labels_lost"],
            "leave_out_busiest_gold_labels_lost_share": loo[
                "gold_labels_lost_share_of_corpus"
            ],
            "community_alpha_spread": a["agreement_by_target"][
                "alpha_spread_across_claimable"
            ],
            "community_baseline_alpha": a["agreement_by_target"][
                "baseline_all_targeted_posts"
            ]["alpha_three_way"],
            "ceiling_three_way": ceiling["ceiling_against_a_human"]["three_way"],
            "ceiling_binary": ceiling["ceiling_against_a_human"]["binary"],
            "mde_test_split_n": mde["test_split_n"],
            "mde_observed_scale_test_split": mde["headline_cell"]["mde_observed_scale"],
            "mde_true_scale_test_split": mde["headline_cell"]["mde_true_scale"],
            "n_required_for_2pt_gap": mde["headline_cell"]["n_required_for_2pt_gap"],
            "gate_blocked_claims": a["signal_gate"]["default_policy"]["blocked_claim_ids"],
            "gate_block_rate": a["signal_gate"]["default_policy"]["summary"]["block_rate"],
        },
        "study_b": {
            "judge_name": b["design"]["judge_name"],
            "exact_match_three_way": b3["judge"]["exact_match"],
            "exact_match_binary": b["binary"]["judge"]["exact_match"],
            "panel_consensus_rate_three_way": b3["judge"]["panel_consensus_rate"],
            "alpha_judge_vs_panel_three_way": b3["judge"][
                "krippendorff_alpha_judge_vs_panel"
            ],
            "unconditioned_match_three_way": b3["judge"][
                "judge_vs_individual_panel_members_unconditioned"
            ]["exact_match"],
            "harness_validation": b["harness_validation"],
            "disattenuated_three_way": b3["disattenuation"][
                "corrected_if_judge_is_deterministic"
            ],
            "constant_baseline_three_way": b3["constant_baseline"]["exact_match"],
            "single_rater_modal_ceiling_three_way": b3["ceiling"][
                "single_rater_modal_ceiling"
            ]["value"],
        },
        "study_c": {
            "single_rater_alpha": proj["single_rater_alpha"],
            "three_rater_reliability": proj["three_rater_reliability"],
            "raters_for_0667": proj["targets"]["0.667"]["raters_required"],
            "raters_for_0800": proj["targets"]["0.800"]["raters_required"],
            "optimum_k_for_power": alloc["optimum_k_at_headline_rate"],
            "k1_true_scale_mde": alloc[
                "is_a_third_rater_worth_more_than_a_third_more_items"
            ]["k1_true_scale_mde"],
            "k3_true_scale_mde": alloc[
                "is_a_third_rater_worth_more_than_a_third_more_items"
            ]["k3_true_scale_mde"],
            "k3_penalty_relative": alloc[
                "is_a_third_rater_worth_more_than_a_third_more_items"
            ]["k3_penalty_relative"],
            "unit_cost_usd_per_label": costs["unit_cost_usd_per_label"],
            "cost_as_published_usd": costs["as_published"]["cost_usd"],
            "cost_for_0800_usd": costs["cost_of_reaching_targets"]["0.800"]["cost_usd"],
            "hybrid_cost_usd": costs["hybrid_design"]["cost_usd"],
        },
        "headlines": {
            "study_a_agreement": a["overall_agreement"]["interpretation"]["statement"],
            "study_a_paradox": a["overall_agreement"]["kappa_paradox"]["conclusion"],
            "study_a_binary": a["binary_contrast"]["reading"],
            "study_a_exposure": a["annotators"]["leave_one_annotator_out"]["reading"],
            "study_a_ceiling": ceiling["ceiling_against_a_human"]["reading"],
            "study_a_mde": mde["reading"],
            "study_a_gate": a["signal_gate"]["reading"],
            "study_b": b["headline"],
            "study_b_ceiling": b3["ceiling"]["headline"],
            "study_c": c["headline"],
        },
    }


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(out)


def render_report(a: dict, b: dict, c: dict, summary: dict) -> str:
    """Markdown summary of all three studies, generated only from the results."""
    battery = a["overall_agreement"]["battery"]
    cons = a["consensus_structure"]["three_way"]
    contrast = a["binary_contrast"]["contrast"]
    parts: list[str] = []

    parts.append("# HateXplain field study -- generated report\n")
    parts.append(
        f"Generated by `rubricon-field` v{__version__} from "
        f"`{a['dataset']['source_path']}`. Every number below is read from "
        "`results/*.json`; none is typed in by hand.\n"
    )
    parts.append(f"> {a['dataset']['citation']}\n")

    parts.append("## Study A -- label-quality audit\n")
    parts.append("### Agreement battery (3-way task, nominal)\n")
    parts.append(
        _table(
            ["coefficient", "value", "n units", "n annotators"],
            [
                ["Krippendorff alpha", battery["krippendorff_alpha"]["value"],
                 battery["krippendorff_alpha"]["n_units"],
                 battery["krippendorff_alpha"]["n_annotators"]],
                ["95% CI (cluster bootstrap)",
                 f"[{battery['ci']['lo']}, {battery['ci']['hi']}]",
                 battery["ci"]["n_clusters"], ""],
                ["Raw pairwise agreement", battery["percent_agreement"]["value"], "", ""],
                ["Fleiss kappa", battery["fleiss_kappa"]["value"], "", ""],
                ["Gwet AC1", battery["gwet_ac1"]["value"], "", ""],
            ],
        )
        + "\n"
    )
    parts.append(a["overall_agreement"]["interpretation"]["statement"] + "\n")
    parts.append("**Kappa paradox test.** " + a["overall_agreement"]["kappa_paradox"]["conclusion"] + "\n")

    parts.append("### Consensus structure\n")
    parts.append(
        _table(
            ["mode", "posts", "share"],
            [[m, cons["counts"][m], f"{cons['percentages'][m]:.1%}"]
             for m in (UNANIMOUS, SPLIT_2_1, NO_MAJORITY)],
        )
        + "\n"
    )
    parts.append(a["consensus_structure"]["reading"] + "\n")

    parts.append("### 3-way versus binary\n")
    parts.append(
        _table(
            ["task", "alpha", "raw agreement", "band"],
            [
                ["3-way", contrast["alpha_three_way"], contrast["raw_three_way"],
                 contrast["band_three_way"]],
                ["binary (toxic vs normal)", contrast["alpha_binary"],
                 contrast["raw_binary"], contrast["band_binary"]],
                ["offensive vs hatespeech (conditional)",
                 contrast["alpha_offensive_vs_hatespeech_conditional"], "", ""],
            ],
        )
        + "\n"
    )
    parts.append(a["binary_contrast"]["reading"] + "\n")

    parts.append("### Annotator load and exposure\n")
    parts.append(
        _table(
            ["annotator", "labels", "share", "alpha delta", "gold labels lost"],
            [
                [r["annotator_id"], r["n_annotations"],
                 f"{r['share_of_all_annotations']:.2%}", r["alpha_delta"],
                 r["gold_labels_lost"]]
                for r in a["annotators"]["leave_one_annotator_out"]["rows"]
            ],
        )
        + "\n"
    )
    parts.append(a["annotators"]["leave_one_annotator_out"]["reading"] + "\n")
    parts.append(a["annotators"]["marginals_ranking"]["reading"] + "\n")

    parts.append("### Agreement by target community\n")
    parts.append(
        _table(
            ["community", "posts", "alpha (3-way)", "raw", "claimable"],
            [
                [r["community"], r["n_posts"], r["alpha_three_way"], r["raw_agreement"],
                 "yes" if r["sufficient_for_claim"] else "no (below floor)"]
                for r in a["agreement_by_target"]["rows"]
            ],
        )
        + "\n"
    )
    parts.append(a["agreement_by_target"]["reading"] + "\n")

    parts.append("### Accuracy ceiling and minimum detectable effect\n")
    parts.append(a["accuracy_ceiling"]["ceiling_against_a_human"]["reading"] + "\n")
    parts.append(
        _table(
            ["n items", "assumed system disagreement", "MDE (observed scale)",
             "MDE (true scale)"],
            [
                [g["n_items"], f"{g['assumed_pairwise_disagreement']:.0%}",
                 g["mde_observed_scale"], g["mde_true_scale"]]
                for g in a["accuracy_ceiling"]["minimum_detectable_effect"]["grid"]
                if g["assumed_pairwise_disagreement"] == A.HEADLINE_DISAGREEMENT_RATE
            ],
        )
        + "\n"
    )
    parts.append(a["accuracy_ceiling"]["minimum_detectable_effect"]["reading"] + "\n")
    parts.append(
        "*"
        + a["accuracy_ceiling"]["minimum_detectable_effect"]["what_this_mde_does_not_cover"]
        + "*\n"
    )

    parts.append("### Signal gate\n")
    parts.append(
        _table(
            ["claim", "kind", "verdict", "reason"],
            [
                [cl["claim_id"], cl["kind"], cl["verdict"].upper(),
                 (cl["blocking_reasons"] or cl["warnings"] or ["--"])[0][:120]]
                for cl in a["signal_gate"]["default_policy"]["claims"]
            ],
        )
        + "\n"
    )
    parts.append(a["signal_gate"]["reading"] + "\n")

    parts.append("## Study B -- judge validation with a held-out human\n")
    parts.append("> " + b["honesty_statement"] + "\n")
    parts.append(b["headline"] + "\n")
    parts.append(
        _table(
            ["measure", "3-way", "binary"],
            [
                ["exact match vs 2-rater panel", b["three_way"]["judge"]["exact_match"],
                 b["binary"]["judge"]["exact_match"]],
                ["panel consensus rate",
                 b["three_way"]["judge"]["panel_consensus_rate"],
                 b["binary"]["judge"]["panel_consensus_rate"]],
                ["alpha judge vs panel",
                 b["three_way"]["judge"]["krippendorff_alpha_judge_vs_panel"],
                 b["binary"]["judge"]["krippendorff_alpha_judge_vs_panel"]],
                ["unconditioned match vs a panel member",
                 b["three_way"]["judge"][
                     "judge_vs_individual_panel_members_unconditioned"]["exact_match"],
                 b["binary"]["judge"][
                     "judge_vs_individual_panel_members_unconditioned"]["exact_match"]],
                ["disattenuated (deterministic judge)",
                 b["three_way"]["disattenuation"]["corrected_if_judge_is_deterministic"],
                 b["binary"]["disattenuation"]["corrected_if_judge_is_deterministic"]],
                ["constant-baseline exact match",
                 b["three_way"]["constant_baseline"]["exact_match"],
                 b["binary"]["constant_baseline"]["exact_match"]],
            ],
        )
        + "\n"
    )
    parts.append("**Ceiling.** " + b["three_way"]["ceiling"]["headline"] + "\n")
    parts.append("**Disattenuation.** " + b["three_way"]["disattenuation"]["reading"] + "\n")
    parts.append(
        "**Harness validation.** "
        + b["harness_validation"]["why_this_is_the_right_check"]
        + f" Corpus pairwise agreement {b['harness_validation']['corpus_pairwise_agreement']}, "
        + f"harness figure {b['harness_validation']['judge_vs_panel_member_unconditioned']}.\n"
    )

    parts.append("## Study C -- replication economics\n")
    parts.append(c["headline"] + "\n")
    parts.append(
        _table(
            ["k raters", "reliability (Spearman-Brown)", "band"],
            [[r["k_raters"], r["reliability"], r["band"]]
             for r in c["reliability_projection"]["projection"]],
        )
        + "\n"
    )
    parts.append(
        _table(
            ["k raters", "items on a fixed budget", "gold reliability",
             "MDE (observed)", "MDE (true scale)"],
            [
                [r["k_raters"], r["n_items"], r["gold_reliability_rho_k"],
                 r["mde_observed_scale"], r["mde_true_scale"]]
                for r in c["budget_allocation"]["grid"]
                if r["assumed_pairwise_disagreement"] == A.HEADLINE_DISAGREEMENT_RATE
            ],
        )
        + "\n"
    )
    parts.append(
        c["budget_allocation"]["is_a_third_rater_worth_more_than_a_third_more_items"][
            "detail"
        ]
        + "\n"
    )
    parts.append("**Why replicate anyway.**\n")
    for reason in c["budget_allocation"]["why_you_should_still_replicate"]:
        parts.append(f"- {reason}")
    parts.append("")
    parts.append(c["cost_model"]["reading"] + "\n")
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"error: dataset not found at {data_path}", file=sys.stderr)
        return 2
    store = Store(Path(args.out))

    print(f"loading {data_path} ...", file=sys.stderr)
    dataset = load_hatexplain(data_path)
    print(
        f"  {dataset.n_posts:,} posts, {dataset.n_annotations:,} annotations, "
        f"{dataset.n_annotators} annotators",
        file=sys.stderr,
    )

    print("study A: label-quality audit ...", file=sys.stderr)
    a = run_study_a(dataset)
    store.write_json("study_a.json", a)

    # Studies B and C recompute alpha at full precision rather than being handed
    # the rounded value out of study A's report dict. The difference is in the
    # fourth decimal place, but a pipeline whose numbers depend on which path
    # they arrived by is a pipeline nobody can reproduce.
    print("study B: judge validation ...", file=sys.stderr)
    b = run_study_b(dataset)
    store.write_json("study_b.json", b)

    print("study C: replication economics ...", file=sys.stderr)
    c = run_study_c(dataset)
    store.write_json("study_c.json", c)

    summary = build_summary(a, b, c)
    store.write_json("summary.json", summary)
    store.write_text("REPORT.md", render_report(a, b, c, summary))

    print(f"wrote {args.out}/study_a.json, study_b.json, study_c.json, "
          f"summary.json, REPORT.md", file=sys.stderr)
    print(json.dumps(summary["headlines"], indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    store = Store(Path(args.results))
    missing = [n for n in ("study_a.json", "study_b.json", "study_c.json")
               if not store.exists(n)]
    if missing:
        print(f"error: missing {', '.join(missing)} in {args.results}; run "
              "`rubricon-field run` first", file=sys.stderr)
        return 2
    a = store.read_json("study_a.json")
    b = store.read_json("study_b.json")
    c = store.read_json("study_c.json")
    summary = store.read_json("summary.json") if store.exists("summary.json") else build_summary(a, b, c)
    text = render_report(a, b, c, summary)
    if args.stdout:
        print(text)
    else:
        store.write_text(args.output, text)
        print(f"wrote {args.results}/{args.output}", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rubricon-field",
        description=(
            "Apply the Rubricon evaluation harness to the HateXplain human "
            "annotations (Mathew et al., AAAI 2021)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"rubricon-field {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run all three studies and write results/")
    p_run.add_argument("--data", default=DEFAULT_DATA, help=f"default: {DEFAULT_DATA}")
    p_run.add_argument("--out", default=DEFAULT_RESULTS, help=f"default: {DEFAULT_RESULTS}")
    p_run.set_defaults(func=cmd_run)

    p_rep = sub.add_parser("report", help="render the markdown summary from results/")
    p_rep.add_argument("--results", default=DEFAULT_RESULTS)
    p_rep.add_argument("--output", default="REPORT.md")
    p_rep.add_argument("--stdout", action="store_true", help="print instead of writing")
    p_rep.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
