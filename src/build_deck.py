"""Build the 7-minute final presentation on the Northwestern template deck, plus the matching speaking script.

    python -m src.build_deck --base Agent_AI_Joint_Updated_7_Minute_Presentation.pptx --curves-dir <dir with the two original-run curve PNGs> \
        --package submission --out SearchLite_RL_Final_Presentation.pptx --script SearchLite_RL_Speaking_Script.md

The base deck supplies the Northwestern layouts and the cover picture; all of its content slides are replaced. Every statistic below is
copied from the Colab output of SearchLite_RL_Final_Fixes.ipynb (cell 4: src.two_comparisons on the raw hard_final JSONs). With --package,
the numbers are re-checked against <package>/tables/two_comparisons.json and the build fails on any mismatch; if the package holds
figures/fig2_m2_grpo_reward_loss.png, an extra slide shows the M2-initialised GRPO curves. Nothing is invented: a figure that is not
available is left out, not drawn.
"""

import argparse
import copy
import json
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

PURPLE, DARK, LILAC, PALE, TEAL, TEAL_BG, GREY, KICK = "4E2A84", "1D1926", "F1E8FA", "FAF8FD", "0D7368", "DDF2ED", "595067", "664396"
ROSE, ROSE_BG, SOFT = "B3261E", "FBE9E7", "E3C7FF"
FONT = "Aptos"
TOTAL_SECONDS = 420

# ---- verified numbers (SearchLite_RL_Final_Fixes.ipynb, cell 4; hard_final, n=120, greedy) -----------------------------------------
N = 120
COUNTS = {"M2 typed_abstain": 34, "M2 typed": 75, "M3 seed 43": 81, "M3 seed 42": 82}
CMP = {  # key in two_comparisons.json -> (both correct, both wrong, model-only, reference-only, diff, ci_lo, ci_hi, p)
    "A: M3_s42 vs M2 (both typed_abstain)": (33, 37, 49, 1, 40.0, 31.7, 49.2, 9.06e-14),
    "B: M3_s42 (typed_abstain) vs M2 (typed)": (74, 37, 8, 1, 5.8, 0.8, 10.8, 0.0391),
    "A: M3_s43 vs M2 (both typed_abstain)": (33, 38, 48, 1, 39.2, 30.8, 48.3, 1.78e-13),
    "B: M3_s43 (typed_abstain) vs M2 (typed)": (74, 38, 7, 1, 5.0, 0.8, 10.0, 0.0703),
    "C: M2 typed vs M2 typed_abstain (same weights)": (33, 44, 42, 1, 34.2, 25.8, 43.3, 1e-11),
}
CATS = ["standard", "distractor_name", "distractor_person", "stale_affiliation", "reverse_two_hop", "unanswerable"]
CAT_LABEL = ["standard", "distractor\nname", "distractor\nperson", "stale\naffiliation", "reverse\ntwo-hop", "unanswer-\nable"]
CAT_COUNTS = {"M2 typed_abstain": [10, 0, 13, 9, 2, 0], "M2 typed": [20, 18, 20, 15, 2, 0], "M3 seed 42": [19, 20, 20, 19, 4, 0]}  # of 20 each


def verify_numbers(package):
    """Re-check every statistic above against the package's two_comparisons.json. Raises AssertionError on any mismatch."""
    res = json.loads((Path(package) / "tables" / "two_comparisons.json").read_text())
    for key, (both, neither, mo, ro, diff, lo, hi, p) in CMP.items():
        c = res["comparisons"][key]
        got = (c["both_correct"], c["both_wrong"], c["model_only_correct"], c["ref_only_correct"])
        assert got == (both, neither, mo, ro), f"{key}: counts {got} != deck {(both, neither, mo, ro)}"
        assert round(c["diff_pts"], 1) == diff and round(c["ci95_pts"][0], 1) == lo and round(c["ci95_pts"][1], 1) == hi, f"{key}: difference/CI"
        assert abs(c["mcnemar_exact_p"] - p) / p < 0.02, f"{key}: p {c['mcnemar_exact_p']} != {p}"
    oc = res["overall_counts"]
    for lab, k in (("M2", 34), ("M2_typed", 75), ("M3_s42", 82), ("M3_s43", 81)):
        assert oc[lab] == [k, N], f"{lab}: {oc[lab]}"
    cc = res["category_counts"]
    for lab, key in (("M2", "M2 typed_abstain"), ("M2_typed", "M2 typed"), ("M3_s42", "M3 seed 42")):
        assert [cc[lab][c][0] for c in CATS] == CAT_COUNTS[key], f"{lab} category counts"
    return True


# ---- drawing helpers -------------------------------------------------------------------------------------------------------------
def rgb(h):
    return RGBColor.from_string(h)


def box(s, x, y, w, h, fill, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08):
    sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill)
    if line:
        sh.line.color.rgb = rgb(line); sh.line.width = Pt(0.75)
    else:
        sh.line.fill.background()
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = radius
    sh.shadow.inherit = False
    return sh


def text(s, x, y, w, h, t, size=12, bold=False, color=DARK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.04); tf.margin_top = tf.margin_bottom = Inches(0.02)
    runs = t if isinstance(t, list) else [[(t, {})]]
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        for seg, o in (para if isinstance(para, list) else [(para, {})]):
            r = p.add_run(); r.text = seg
            r.font.name = FONT; r.font.size = Pt(o.get("size", size)); r.font.bold = o.get("bold", bold); r.font.italic = o.get("italic", italic)
            r.font.color.rgb = rgb(o.get("color", color))
    return tb


def header(s, kicker, title, sub, n):
    text(s, 0.65, 0.26, 8.85, 0.26, kicker, 9.5, True, KICK)
    text(s, 0.63, 0.62, 8.9, 0.75, title, 24, True, PURPLE, anchor=MSO_ANCHOR.TOP)
    text(s, 0.66, 1.4, 8.8, 0.4, sub, 12, False, GREY)
    ln = s.shapes.add_connector(1, Inches(0.65), Inches(1.9), Inches(9.35), Inches(1.9))
    ln.line.color.rgb = rgb("D9D2E3"); ln.line.width = Pt(1.25)
    text(s, 9.31, 7.12, 0.3, 0.22, str(n), 9, False, "FFFFFF")


def card(s, x, y, w, h, label, body, fill=LILAC, lab_color=PURPLE, body_size=12, lab_size=10.5, body_color=DARK, bold_body=False):
    box(s, x, y, w, h, fill)
    text(s, x + 0.15, y + 0.08, w - 0.3, 0.28, label, lab_size, True, lab_color)
    text(s, x + 0.15, y + 0.38, w - 0.3, h - 0.45, body, body_size, bold_body, body_color)


def footer_note(s, t, y=6.68):
    text(s, 0.7, y, 8.6, 0.28, t, 9, False, GREY, align=PP_ALIGN.CENTER)


def style_axis_text(chart, size=10):
    chart.font.size = Pt(size); chart.font.name = FONT; chart.font.color.rgb = rgb(DARK)


def bar_chart(s, x, y, w, h, cats, series, colors, vmax, number_format='0.0"%"', legend=False, gap=60, point_colors=None, label_size=11, overlap=None):
    cd = CategoryChartData(); cd.categories = cats
    for name, vals in series:
        cd.add_series(name, vals)
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart; style_axis_text(ch); ch.has_title = False
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(10)
    va = ch.value_axis; va.maximum_scale = vmax; va.minimum_scale = 0; va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb("E6E0EE"); va.format.line.fill.background()
    va.tick_labels.font.size = Pt(9); ch.category_axis.tick_labels.font.size = Pt(10)
    ch.category_axis.format.line.color.rgb = rgb("9A90A8")
    pl = ch.plots[0]; pl.gap_width = gap; pl.has_data_labels = True
    if overlap is not None:
        pl.overlap = overlap
    dl = pl.data_labels; dl.number_format = number_format; dl.number_format_is_linked = False
    dl.font.size = Pt(label_size); dl.font.bold = True; dl.position = XL_LABEL_POSITION.OUTSIDE_END
    for i, ser in enumerate(pl.series):
        ser.format.fill.solid(); ser.format.fill.fore_color.rgb = rgb(colors[i]); ser.invert_if_negative = False
    if point_colors:
        for j, c in enumerate(point_colors):
            pt = pl.series[0].points[j]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = rgb(c)
    return ch


def notes(s, seconds, say, background):
    s.notes_slide.notes_text_frame.text = (f"SPEAK ALOUD — TARGET {seconds} SECONDS (part of 7:00 total)\n\n{say}\n\n"
                                           f"────────────────────────────────────\nOPTIONAL Q&A / BACKGROUND — DO NOT READ ALOUD\n\n{background}")


def pct(k, n=N):
    return 100 * k / n


# ---- slides ------------------------------------------------------------------------------------------------------------------------
def build(base, curves_dir, package, out, script_out):
    prs = Presentation(base)
    lay = prs.slide_layouts.get_by_name("Title Slide")
    # drop all content slides, keep the cover (slide 1: full-bleed Northwestern cover picture + text boxes)
    ids = prs.slides._sldIdLst
    for sld in list(ids)[1:]:
        prs.part.drop_rel(sld.rId); ids.remove(sld)
    cover = prs.slides[0]
    tbs = [sh for sh in cover.shapes if sh.has_text_frame]
    for sh, t in zip(tbs, ["Can RL teach a small model to search?", "And how much of the gain is RL, rather than prompt design?",
                           "GRPO on Qwen3-0.6B + LoRA  ·  BM25 search environment", "CS 496 · Agent AI  |  Solo pilot project  |  November 9, 2026"]):
        for r in sh.text_frame.paragraphs[0].runs[1:]:
            r._r.getparent().remove(r._r)
        sh.text_frame.paragraphs[0].runs[0].text = t
    script = []
    cover_say = ("Can reinforcement learning teach a small language model to use a search tool? I trained Qwen3-0.6B with GRPO in a controlled search "
                 "environment, and I measured a large gain. But the more interesting question, which I will answer with paired statistics, is how much "
                 "of that gain is really reinforcement learning, and how much is just prompt design.")
    notes(cover, 20, cover_say, "Pilot project for CS 496 Agent AI. Everything shown is from saved evaluation files; nothing was retrained for this presentation. "
                                 "Sealed benchmark: hard_final, 120 questions, one evaluation per model after the model was frozen (M3 seed 42).")
    script.append(("1. Cover", 20, cover_say))
    n = [1]

    def new(kicker, title, sub):
        s = prs.slides.add_slide(lay)
        for ph in list(s.placeholders):
            ph._element.getparent().remove(ph._element)
        n[0] += 1
        header(s, kicker, title, sub, n[0])
        return s

    # 2 ---- framework (rubric 1)
    s = new("REQUIREMENT 1 · A WORKING RL FRAMEWORK", "A search agent in a controlled world", "Qwen3-0.6B + LoRA · BM25 over a synthetic corpus · deterministic reward")
    box(s, 0.78, 2.1, 8.45, 0.62, LILAC)
    text(s, 0.95, 2.18, 8.1, 0.5, "QUESTION  “Where does the lead researcher of Project X work?”", 14, True, DARK, anchor=MSO_ANCHOR.MIDDLE)
    box(s, 0.78, 2.9, 8.45, 0.78, PURPLE)
    text(s, 0.95, 2.96, 8.1, 0.66, [[("POLICY  Qwen3-0.6B + LoRA  chooses an action", {"bold": True, "size": 14.5, "color": "FFFFFF"})],
                                    [("search(query)  →  BM25 returns top-2 records (max 2 searches)       or       <answer>…</answer>", {"size": 12, "color": SOFT})]],
         anchor=MSO_ANCHOR.MIDDLE)
    box(s, 0.78, 3.86, 4.1, 1.0, TEAL_BG); box(s, 5.13, 3.86, 4.1, 1.0, LILAC)
    text(s, 0.93, 3.92, 3.8, 0.92, [[("MDP", {"bold": True, "size": 10.5, "color": TEAL})], [("state: question + records so far · action: search or answer · reward only at the end", {"size": 11.5})]])
    text(s, 5.28, 3.92, 3.8, 0.92, [[("VERIFIER (deterministic)", {"bold": True, "size": 10.5, "color": PURPLE})],
                                    [("R = 1.0·correct − 0.05·searches − 0.10·invalid", {"size": 12.5, "bold": True})]])
    box(s, 0.78, 5.04, 8.45, 0.72, DARK)
    text(s, 0.95, 5.1, 8.1, 0.6, [[("M2 = SFT: imitate example traces      →      M3 = GRPO on top of M2", {"bold": True, "size": 14, "color": "FFFFFF"})],
                                  [("GRPO samples a group of attempts per question and scores each against the group mean: no value network", {"size": 11, "color": SOFT})]],
         anchor=MSO_ANCHOR.MIDDLE)
    for i, (x, w, t) in enumerate([(0.78, 2.0, "TRL 1.17 GRPO"), (2.95, 2.3, "LoRA r=8, q/v proj"), (5.42, 1.9, "free Colab L4"), (7.5, 1.75, "β = 0 (no KL)")]):
        box(s, x, 5.98, w, 0.36, LILAC); text(s, x, 6.01, w, 0.3, t, 10.5, True, PURPLE, align=PP_ALIGN.CENTER)
    footer_note(s, "Hard benchmark: 6 question categories · hard_train 120 · hard_dev 60 · hard_final 120 (sealed, evaluated once after the freeze)", 6.5)
    say = ("This is the framework, and it runs end to end. The environment is a small Markov decision process. The state is the question plus the records "
           "retrieved so far. At each step the agent, Qwen3-0.6B with LoRA adapters, either searches, where BM25 returns the top two records and at most "
           "two searches are allowed, or it gives an answer. A deterministic verifier scores the whole episode: one point for a correct answer, minus 0.05 per "
           "search, minus 0.10 per invalid action. I compare two ways of training. Supervised fine-tuning imitates example traces; that is model M2. "
           "GRPO then continues from M2: it samples a group of attempts for each question and scores each one against the group average, so it needs no value network. "
           "That is model M3.")
    notes(s, 55, say, "MDP: transitions are the BM25 results appended to the context; episode ends on answer or after 2 searches. Reward is verified by exact normalised match. "
                      "GRPO advantage = (r − mean)/(std + 1e-4) per group; groups with identical rewards give zero gradient. beta=0 so no KL term. "
                      "Config: configs/train_hard_grpo.yaml (LoRA r=8 on q_proj,v_proj; lr 2e-5; temperature 0.7 unless overridden by the signal probe). "
                      "Hard categories: standard, distractor_name, distractor_person, stale_affiliation, reverse_two_hop, unanswerable.")
    script.append(("2. Framework (Requirement 1)", 55, say))

    # 3 ---- original-run curves (rubric 2)
    s = new("REQUIREMENT 2 · REWARD AND LOSS CURVES", "What happened during training?", "Original from-base GRPO run, 60 optimizer steps · real per-step logs")
    c1, c2 = Path(curves_dir) / "curve1_reward.png", Path(curves_dir) / "curve2_loss.png"
    s.shapes.add_picture(str(c1), Inches(0.54), Inches(2.1), Inches(4.5)); s.shapes.add_picture(str(c2), Inches(4.93), Inches(2.1), Inches(4.5))
    card(s, 0.78, 5.1, 4.2, 1.4, "REWARD: NOISY BUT GENERALLY RISING", "Each point is one batch of sampled attempts, so single steps swing. Judge the trend, not one step.", LILAC)
    card(s, 5.12, 5.1, 4.1, 1.4, "LOSS NEAR ZERO ≠ CONVERGENCE", "GRPO loss is a relative-advantage objective, not prediction error. It is ~0 whenever a group’s attempts tie.", LILAC)
    footer_note(s, "Light line: each step · dark line: 7-step average. From the original run’s logged steps; its evaluation is reported separately.", 6.62)
    say = ("These are the real training curves from the original GRPO run: reward on the left, loss on the right. Reward is the average score of the sampled attempts at each "
           "update. It is noisy, because every batch contains different questions and different samples, but the seven-step average trends upward. The loss stays around zero, "
           "and I want to be careful here: that is not an error rate and it is not evidence of convergence. GRPO compares attempts within a group, and when every attempt "
           "in a group gets the same reward there is no advantage and no gradient. So I judge learning by held-out behaviour, not by the loss curve.")
    notes(s, 45, say, "Loss = token-averaged negative clipped-ratio × advantage, beta=0. At step 1 the ratio is exactly 1 so loss≈0 by construction. Loss is exactly 0 on a step only when every group "
                      "in the batch has tied rewards. The report packages the full CSVs (recovered unchanged from Drive for the original run) and Figure 2 shows the same curves for the M2-initialised runs "
                      "(M3, M4, M4g). This figure is the original run presented earlier in the pilot; Figure 1 in the report is regenerated from the recovered log.")
    script.append(("3. Reward and loss curves (Requirement 2)", 45, say))

    fig2 = Path(package) / "figures" / "fig2_m2_grpo_reward_loss.png" if package else None
    if fig2 and fig2.exists():
        s = new("REQUIREMENT 2 · REWARD AND LOSS CURVES", "Training from the SFT checkpoint (M2 → M3)", "GRPO on hard_train, original reward · real per-step logs from the packaged runs")
        s.shapes.add_picture(str(fig2), Inches(0.6), Inches(2.05), Inches(8.8))
        footer_note(s, "Figure 2 of the report. Reward variation inside groups, not the loss value, decides whether GRPO can learn.", 6.62)
        say = ("The same two views for the runs that start from the supervised checkpoint, which is the setting that produced the headline model. Again, the loss tells us little on its own; "
               "what matters is whether rewards differ within a group, because that is what creates a learning signal.")
        notes(s, 25, say, "Figure 2 from submission/figures. Reward-variation figure (fig3) in the report shows the fraction of zero-spread groups.")
        script.append(("3b. Curves from the SFT start", 25, say))

    # 4 ---- headline
    s = new("RESULTS · SEALED hard_final (n = 120)", "Fixed prompt: 28.3% → 68.3%", "Same evaluation prompt (typed_abstain), greedy decoding, 120 held-out questions · frozen model: M3 seed 42")
    labs = ["M2 SFT\n(typed_abstain)", "M2 SFT\n(typed)", "M3 GRPO\nseed 43", "M3 GRPO\nseed 42 (frozen)"]
    vals = [round(pct(COUNTS[k]), 1) for k in ("M2 typed_abstain", "M2 typed", "M3 seed 43", "M3 seed 42")]
    bar_chart(s, 0.55, 2.05, 5.3, 4.1, labs, [("accuracy", vals)], [PURPLE], 100, point_colors=["B9A6D6", "8D6FB8", "6F4BA3", PURPLE], gap=45, label_size=12)
    box(s, 6.05, 2.15, 3.2, 1.7, DARK)
    text(s, 6.15, 2.22, 3.0, 0.5, "A · FIXED PROMPT", 10.5, True, SOFT, align=PP_ALIGN.CENTER)
    text(s, 6.15, 2.55, 3.0, 0.6, "+40.0 pts", 28, True, "FFFFFF", align=PP_ALIGN.CENTER)
    text(s, 6.15, 3.2, 3.0, 0.6, "34/120 → 82/120 · 95% CI [+31.7, +49.2]", 10.5, False, SOFT, align=PP_ALIGN.CENTER)
    card(s, 6.05, 4.0, 3.2, 2.15, "PAIRED (SAME 120 QUESTIONS)", [[("49", {"bold": True, "size": 15, "color": TEAL}), (" fixed by M3 only", {"size": 11})],
                                                                [("1", {"bold": True, "size": 15, "color": ROSE}), (" lost by M3 only", {"size": 11})],
                                                                [("33 both right · 37 both wrong", {"size": 11})], [("McNemar exact p = 9.1e-14", {"size": 11, "bold": True})]], LILAC)
    footer_note(s, "Bars: accuracy on the 120 sealed questions (counts 34, 75, 81, 82). Seed 43 (replication): A = +39.2 pts [+30.8, +48.3].", 6.45)
    say = ("Here is the headline on 120 sealed questions that were never used for training or model selection. Under the same evaluation prompt, the supervised model M2 gets 34 of 120, "
           "28.3 percent. The frozen GRPO model, M3 seed 42, gets 82 of 120, 68.3 percent. That is plus 40 points, with a paired confidence interval from about 32 to 49, "
           "and 49 questions fixed by GRPO against one lost. A second training seed gives 67.5 percent, so it is not a one-seed accident. "
           "But look at the second bar: the very same supervised weights score 62.5 percent when I only change the prompt reminder.")
    notes(s, 40, say, "M3 seed 42 was fixed by a pre-declared selection rule (src/freeze_hard.py) before hard_final was evaluated; seed 43 is a replication, not a choice. "
                      "Paired CI = bootstrap of the accuracy difference over question ids (5000 resamples, seed 0). McNemar exact two-sided = exact sign test on the 49 vs 1 discordant questions. "
                      "Sealed-set hygiene: FROZEN_HARD.json, adapter sha256 eda95b43…, verified in tables/freeze_verification.md.")
    script.append(("4. Headline result", 40, say))

    # 5 ---- attribution
    s = new("HOW MUCH IS RL, HOW MUCH IS THE PROMPT?", "Most of the +40 appears with the prompt alone", "Same SFT weights, only the answer-format reminder changes · then the stronger baseline vs GRPO")
    card(s, 0.78, 2.1, 2.7, 1.75, "C · PROMPT ALONE (NO RL)", [[("+34.2 pts", {"bold": True, "size": 22, "color": ROSE})], [("M2: 28.3% → 62.5%", {"size": 11.5})],
                                                               [("CI [+25.8, +43.3] · p = 1e-11", {"size": 10})]], ROSE_BG, ROSE)
    card(s, 3.65, 2.1, 2.7, 1.75, "A · FIXED PROMPT", [[("+40.0 pts", {"bold": True, "size": 22, "color": PURPLE})], [("M2 28.3% → M3 68.3%", {"size": 11.5})],
                                                       [("CI [+31.7, +49.2] · p = 9e-14", {"size": 10})]], LILAC)
    card(s, 6.52, 2.1, 2.7, 1.75, "B · STRONGER BASELINE", [[("+5.8 pts", {"bold": True, "size": 22, "color": TEAL})], [("M2 (typed) 62.5% → M3 68.3%", {"size": 11.5})],
                                                           [("CI [+0.8, +10.8] · p = 0.039", {"size": 10})]], TEAL_BG, TEAL)
    box(s, 0.78, 4.05, 8.45, 1.85, PALE, line="D9D2E3")
    text(s, 0.95, 4.12, 8.15, 1.75, [[("Read it carefully.", {"bold": True, "size": 13, "color": PURPLE})],
                                    [("• C is the same weights, no RL: only the reminder changed.", {"size": 12.5})],
                                    [("• B: 8 questions won vs 1 lost; the interval barely excludes 0. Seed 43 gives +5.0 (p = 0.070): not significant.", {"size": 12.5})],
                                    [("• B is not a clean RL estimate: M3 also trained on hard_train, M2 did not, and there is no SFT-on-hard_train control.", {"size": 12.5})]])
    box(s, 0.78, 6.05, 8.45, 0.5, DARK)
    text(s, 0.95, 6.08, 8.15, 0.44, "Honest claim: GRPO adds a modest, borderline-significant gain beyond a good prompt; the SFT model is strongly prompt-sensitive.", 12, True, "FFFFFF", anchor=MSO_ANCHOR.MIDDLE)
    say = ("So how much is RL? I ran two comparisons. Comparison A holds the prompt fixed: plus 40 points. But when I keep the same supervised weights and only give them the reminder they were "
           "trained with, accuracy jumps from 28.3 to 62.5 percent, plus 34 points, with no reinforcement learning at all. Comparison B uses that stronger baseline: M3 at 68.3 versus 62.5, "
           "plus 5.8 points. That interval just clears zero, with 8 questions won and 1 lost, and for the second seed the gain is 5.0 points and not significant. "
           "Also, M3 trained on extra hard training questions that M2 never saw, and I have no supervised control on that data, so even B is not a pure RL effect. "
           "The honest claim is: most of the headline is prompt design, and GRPO adds a smaller gain on top.")
    notes(s, 60, say, "Arithmetic decomposition only: 34.2 of the 40.0 points appear from the reminder change alone and 5.8 remain; these are not additive causal shares. "
                      "B settings differ in answer_format only (typed vs typed_abstain); A has identical settings except the checkpoint. Seed 43: B diff +5.0, CI [+0.8,+10.0], 7 vs 1, p=0.0703. "
                      "Confounds: (1) M2 evaluated under typed_abstain, a reminder it never trained with; (2) GRPO runs trained on hard_train under typed_abstain; (3) no SFT-on-hard_train control; (4) no GRPO run under typed. "
                      "Follow-up that would separate them: SFT on hard_train, and GRPO under typed. Not run here (no new experiments).")
    script.append(("5. How much is RL vs prompt design", 60, say))

    # 6 ---- categories
    s = new("WHERE IT WORKS, WHERE IT FAILS", "Gains are uneven; two categories stay broken", "Correct answers out of 20 per category · sealed hard_final · M3 seed 42 vs both SFT baselines")
    ch = bar_chart(s, 0.5, 2.0, 9.0, 3.2, CAT_LABEL, [(k, CAT_COUNTS[k]) for k in ("M2 typed_abstain", "M2 typed", "M3 seed 42")], ["B9A6D6", "8D6FB8", PURPLE], 22,
                   number_format="0", legend=True, gap=70, label_size=10, overlap=-5)
    ch.value_axis.major_unit = 5
    card(s, 0.78, 5.32, 4.15, 1.25, "REVERSE TWO-HOP: 4/20", "Needs sequential retrieval: the second query depends on the first result. SFT 2/20 → M3 4/20.", ROSE_BG, ROSE, body_size=11)
    card(s, 5.08, 5.32, 4.15, 1.25, "UNANSWERABLE: 0/20", "None of the three models gave a correct abstention (M3 also 0/10 on dev). Trace labels: report §15.", ROSE_BG, ROSE, body_size=11)
    say = ("Per category, the story is uneven. Distractor-name questions go from zero to twenty out of twenty under the fixed prompt, and stale-affiliation from 9 to 19. "
           "But two categories stay broken. Reverse two-hop questions, where the second search depends on what the first returned, are only 4 of 20: GRPO did not teach reliable sequential retrieval. "
           "And on unanswerable questions every model scores zero of twenty: none produced a correct abstention. "
           "An overall 68 percent hides that. The report classifies the real traces for these two categories.")
    notes(s, 40, say, "M3 seed 42 per category: standard 19, distractor_name 20, distractor_person 20, stale_affiliation 19, reverse_two_hop 4, unanswerable 0 (of 20). "
                      "M2 typed_abstain: 10, 0, 13, 9, 2, 0. M2 typed: 20, 18, 20, 15, 2, 0. Against the typed baseline the M3 gains are only in distractor_name (+2), stale_affiliation (+4), reverse_two_hop (+2); standard is −1. "
                      "20 questions per category: a one-question change is 5 points, so category-level claims are descriptive. Failure labels come from src/trace_facts.py applied to saved traces.")
    script.append(("6. Per-category results and failures", 40, say))

    # 7 ---- case studies A/B
    s = new("REQUIREMENT 3 · CASE STUDIES (REAL TRACES)", "Search, or answer immediately?", "Real traces from the one-hop pilot evaluation: observable tool actions, not hidden reasoning")
    for x, kick, q, act, det, res in [(0.73, "CASE A · SEARCH REQUIRED", "“Which institution is Omar Kafen affiliated with?”", "ACTION: search once",
                                       "BM25 retrieved: “Omar Kafen is affiliated with Ethlotri Council…”", "✓ Ethlotri Council  ·  reward 0.95"),
                                      (5.1, "CASE B · NO SEARCH REQUIRED", "“Project Melcorris is funded by Tharrislo Laboratory. Which agency funds it?”", "ACTION: zero searches",
                                       "The answer was already in the question.", "✓ Tharrislo Laboratory  ·  reward 1.00")]:
        box(s, x, 2.15, 4.13, 3.75, PALE, line="D9D2E3"); box(s, x, 2.15, 4.13, 0.46, PURPLE)
        text(s, x + 0.15, 2.22, 3.8, 0.3, kick, 12, True, "FFFFFF")
        text(s, x + 0.17, 2.78, 3.8, 1.0, q, 15, True, DARK)
        text(s, x + 0.17, 3.85, 3.8, 0.3, act, 12, True, PURPLE)
        text(s, x + 0.17, 4.2, 3.8, 0.7, det, 12.5, False, DARK)
        box(s, x + 0.2, 5.05, 3.7, 0.6, TEAL_BG); text(s, x + 0.3, 5.1, 3.5, 0.5, res, 12, True, TEAL, anchor=MSO_ANCHOR.MIDDLE)
    box(s, 0.78, 6.05, 8.45, 0.5, DARK)
    text(s, 0.95, 6.08, 8.15, 0.44, "A good tool-using agent acts when needed and avoids unnecessary action.  Reward = 1·correct − 0.05·searches.", 12, True, "FFFFFF", anchor=MSO_ANCHOR.MIDDLE)
    say = ("Now the case studies, taken from real saved traces, not from the model's private reasoning. On the left, the question asks where Omar Kafen works, which the question does not say. "
           "The agent searches once, BM25 returns the right record, and it answers Ethlotri Council for a reward of 0.95: one point for being right, minus 0.05 for the search. "
           "On the right, the funder is already in the question. The agent answers directly with no search and earns 1.00. That is the behaviour the reward is designed to encourage: search only when it helps.")
    notes(s, 35, say, "Both are fictional test entities from the original one-hop pilot evaluation (test_lookup_001 and test_no_search_000), verified against the test corpus. "
                      "R = 1 correct − 0.05 searches − 0.10 invalid. Non-thinking model; never claim the model revealed reasoning. Additional hard-benchmark traces (distractor, stale-affiliation, reverse two-hop, unanswerable) are in submission/case_studies.md and report §13.")
    script.append(("7. Case studies A and B (Requirement 3)", 35, say))

    # 8 ---- two-hop + exploration
    s = new("BONUS · EXPLORATION AND WHAT RL CANNOT CREATE", "RL sharpens a behaviour; it did not invent one", "Two-hop questions need a second search. Where did it come from?")
    box(s, 0.78, 2.1, 4.15, 2.0, PALE, line="D9D2E3")
    text(s, 0.93, 2.15, 3.9, 1.95, [[("REAL TWO-HOP TRACE", {"bold": True, "size": 10.5, "color": PURPLE})],
                                    [("“Where does the lead researcher of Project Serethvel work?”", {"bold": True, "size": 12})],
                                    [("search 1: Project → Priya Ristav", {"size": 12.5})], [("needed: Priya → Kabraol Laboratory", {"size": 12.5})],
                                    [("Original GRPO agent stopped after search 1 → wrong", {"size": 12.5, "bold": True, "color": ROSE})]])
    card(s, 5.08, 2.1, 4.15, 2.0, "TWO-HOP DEV (30 QUESTIONS)", [[("Original GRPO (no SFT): 0/30", {"size": 13})], [("RL-only, F0 (no SFT): 0/30", {"size": 13})],
                                                                  [("M2 (SFT): 14/30", {"size": 13})], [("M3 (SFT → GRPO): 29/30", {"size": 14, "bold": True, "color": TEAL})]], TEAL_BG, TEAL)
    card(s, 0.78, 4.28, 4.15, 2.2, "EXPLORATION FAILED WITHOUT SFT", "Base model sampled a second search 0/80 times at T=0.7 and 0/80 at T=1.0. With no reward difference inside a group, GRPO has nothing to reinforce. Two exploration-incentive pilots (F2p, F3) never sampled a second search.", ROSE_BG, ROSE, body_size=12.5)
    card(s, 5.08, 4.28, 4.15, 2.2, "FRAGILE EVEN WITH SFT START", "Partial-SFT start, original reward: seed 42 reached 30/30, but seed 43 collapsed to never searching (0/30). Earlier run: reward rose while lookup accuracy fell 18.3% → 0% (searches per question 0.55 → 0).", LILAC, body_size=12.5)
    say = ("Why did GRPO work here at all? Look at two-hop questions. In this real trace the question needs two searches, and the original agent stopped after the first and answered with an intermediate name. "
           "The base model never sampled a second search, zero out of 160 tries, so GRPO had no better attempt to reinforce, and my RL-only pilots with extra exploration incentives never found one either. "
           "I do not claim GRPO discovered two-hop search on its own. What worked was starting from a supervised checkpoint that already searched twice some of the time: M2 gets 14 of 30 on two-hop dev, "
           "and GRPO lifts that to 29 of 30. Even then it is fragile: one partial-SFT seed collapsed to never searching, and an earlier run raised reward while lookup accuracy fell to zero. "
           "So reward variation, exploration and seed all matter.")
    notes(s, 55, say, "Sources: base-model second-search probe 0/80 (T=0.7), 0/80 (T=1.0); F0 0/30 under both reminders; F2p and F3 pilots produced no two-hop rollout with a second search so their full runs did not start (pre-declared gate); "
                      "E1 (partial-SFT, original reward) seed 42 30/30 vs seed 43 0/30 on twohop_dev; E2 (evidence-aware) 30/30 both seeds. Earlier shortcut run: overall 37%→38%, lookup 18.3%→0%, searches/question 0.55→0 "
                      "(a different setup from the later runs; do not compare their baselines). Twohop-dev M2 14/30 → M3 29/30 on development data, not sealed. The slide’s trace is from the pilot’s original GRPO model.")
    script.append(("8. Exploration failure and the role of SFT", 55, say))

    # 9 ---- takeaways
    s = new("TAKEAWAYS AND LIMITS", "What this pilot supports, and what it does not", "A rigorous result is a bounded result")
    box(s, 0.78, 2.1, 8.45, 1.05, DARK)
    text(s, 0.95, 2.14, 8.15, 0.98, "RL can improve a search agent that already has incomplete but useful tool-use behaviour. Insufficient exploration, sparse reward variation and training instability can prevent learning.", 13.5, True, "FFFFFF", anchor=MSO_ANCHOR.MIDDLE)
    card(s, 0.78, 3.33, 4.15, 2.0, "SUPPORTED", [[("• +40.0 pts under a fixed prompt, both seeds", {"size": 12.5})], [("• +5.8 pts (p = 0.039) vs the stronger SFT baseline; seed 43 not significant", {"size": 12.5})],
                                                [("• Working framework, real curves, real traces", {"size": 12.5})]], TEAL_BG, TEAL)
    card(s, 5.08, 3.33, 4.15, 2.0, "NOT SUPPORTED", [[("• “GRPO beats SFT”: no SFT-on-hard_train control", {"size": 12.5})], [("• GRPO discovering two-hop search alone", {"size": 12.5})],
                                                     [("• Convergence from a near-zero loss", {"size": 12.5})]], ROSE_BG, ROSE)
    box(s, 0.78, 5.5, 8.45, 1.05, PALE, line="D9D2E3")
    text(s, 0.95, 5.54, 8.15, 0.98, [[("Limits: ", {"bold": True, "size": 11, "color": PURPLE}),
                                      ("0.6B model · synthetic fixed-template corpora · BM25 with ≤2 searches · 20 questions per category · 2 seeds per variant · greedy decoding, one sealed evaluation. "
                                       "Generalisation to real corpora, longer chains or other prompts is untested. Next: SFT on hard_train and GRPO under the typed prompt.", {"size": 11})]], anchor=MSO_ANCHOR.MIDDLE)
    say = ("To close. What this pilot supports: GRPO can improve a search agent that already has incomplete but useful tool-use behaviour. I measured plus 40 points under a fixed prompt, on two seeds, "
           "and a smaller, borderline-significant plus 5.8 against the stronger supervised baseline. What it does not support: that GRPO beats supervised learning, because I have no supervised control on the same data; "
           "that GRPO discovers multi-step search by itself; or that a near-zero loss means convergence. The limits are real: a 0.6-billion-parameter model, synthetic corpora, 20 questions per category, "
           "two seeds, and prompt sensitivity that I only partly controlled. The next experiments are SFT on the hard training data and GRPO under the stronger prompt. Thank you.")
    notes(s, 45, say, "Rubric map: (1) working RL framework: slides 2 and 4; (2) reward and loss curves: slide 3 (+3b if packaged); (3) case studies: slides 7 and 8. Bonus: paired A/B analysis, exploration/partial-SFT/RL-only studies, prompt-sensitivity audit. "
                      "Package: README_SUBMISSION.md, FINAL_PILOT_REPORT.md, FROZEN_HARD.json, MANIFEST_SHA256.txt. No new experiments were run for this deck.")
    script.append(("9. Takeaways and limits", 45, say))

    prs.save(out)
    total = sum(sec for _, sec, _ in script)
    words = sum(len(t.split()) for _, _, t in script)
    lines = ["# SearchLite-RL final presentation: speaking script", "",
             f"Target: 7:00. Slide timings sum to {total // 60}:{total % 60:02d}; the script is {words} words (about {words / 150:.1f} min at 150 words per minute). Read each section aloud; the "
             "same text is in each slide's speaker notes.", ""]
    for title, sec, t in script:
        lines += [f"## {title}  ({sec} s)", "", t, ""]
    Path(script_out).write_text("\n".join(lines))
    return len(prs.slides), total, words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--curves-dir", required=True)
    ap.add_argument("--package", default=None)
    ap.add_argument("--out", default="SearchLite_RL_Final_Presentation.pptx")
    ap.add_argument("--script", default="SearchLite_RL_Speaking_Script.md")
    args = ap.parse_args()
    if args.package and (Path(args.package) / "tables" / "two_comparisons.json").exists():
        verify_numbers(args.package)
        print("numbers verified against", Path(args.package) / "tables" / "two_comparisons.json")
    elif args.package:
        print("WARNING: no two_comparisons.json in the package; numbers are the deck's built-in values from the Colab output")
    n, total, words = build(args.base, args.curves_dir, args.package, args.out, args.script)
    print(f"wrote {args.out}: {n} slides, timings {total}s, script {words} words")


if __name__ == "__main__":
    main()
