/* ============================================================
   Paper corpus for the immersive "visual thesis" walkthroughs.
   Rendered by paper.html via ?id=<key>.
   ORDER defines the narrative thread (prev/next navigation).
   All quantitative claims are drawn from the papers themselves.
   ============================================================ */

const ME = "Wisdom O. Ikezogwo";

// Bottom-to-top rungs of the research stack (mirrors index.html vision).
const RUNGS = {
  perception:  { label: "Perception",            accent: "#1772d0" },
  grounding:   { label: "Grounding",             accent: "#2a9d8f" },
  reasoning:   { label: "Reasoning & Agents",    accent: "#7048c4" },
  rl:          { label: "Self-Improvement (RL)", accent: "#c9791b" },
  embodiment:  { label: "Physics & Embodiment",  accent: "#c0392b" },
  origins:     { label: "Origins",               accent: "#607080" },
};

const ORDER = [
  "quilt1m", "quilt-llava", "mednarratives", "medblink",
  "pathfinder", "rubrics", "svg2", "mscamil", "mmae", "eeg-dg"
];

const PAPERS = {

  /* ---------------------------------------------------------- */
  "quilt1m": {
    title: "Quilt-1M: One Million Image–Text Pairs for Histopathology",
    authors: `<b>${ME}</b>, Mehmet Saygin Seyfioglu, Fatemeh Ghezloo, Dylan Geva, Fatwir S. Mohammed, Pavan K. Anand, Ranjay Krishna, Linda Shapiro`,
    venue: "NeurIPS 2023", award: "Oral", year: 2023,
    rung: "perception",
    tagline: "If natural-image AI runs on web-scale image–text data, where does pathology's data come from? It was hiding in plain sight — narrated by experts on YouTube.",
    teaser: "images/quilt1m.png",
    links: { arxiv: "https://arxiv.org/abs/2306.11207", pdf: "https://arxiv.org/pdf/2306.11207", web: "https://quilt1m.github.io/" },
    embed: "https://arxiv.org/pdf/2306.11207",
    beats: [
      { kind: "problem", heading: "The data desert",
        text: "Vision–language models thrive on hundreds of millions of paired image–text examples. Histopathology had almost none at scale — the largest comprehensive dataset held roughly 200K pairs, capping every model built on top of it." },
      { kind: "insight", heading: "Experts are already narrating",
        text: "Thousands of hours of educational pathology exist on YouTube, where expert narrators describe exactly what they see as they see it. The spoken word, aligned to the frame on screen, is a naturally occurring image–text pair.",
        figure: "images/pipeline_clustering.png",
        caption: "A curation pipeline pairs extracted frames with transcribed narration." },
      { kind: "method", heading: "A curation pipeline, not a scrape",
        text: "We combine large language models, handcrafted algorithms, human knowledge databases, and automatic speech recognition to denoise transcripts and align them to frames — then augment the YouTube corpus with pairs from Twitter, research papers, and the wider web." },
      { kind: "result", heading: "5× the field, and a better model",
        text: "1,087 hours of video yielded 802K pairs from YouTube alone, expanded to one million total — about 5× the largest prior comprehensive dataset. A CLIP model fine-tuned on Quilt-1M (QuiltNet) set a new state of the art on zero-shot and linear-probe classification across 13 patch-level datasets spanning 8 sub-pathologies, plus cross-modal retrieval." },
      { kind: "why", heading: "The floor of the stack",
        text: "You cannot ground, reason about, or act on tissue a model has never learned to see. Quilt-1M supplies the scale of grounded visual–linguistic supervision that everything above it — grounding, agents, RL — is built on." },
    ],
    stats: [
      { value: "1M", label: "image–text pairs" },
      { value: "1,087 hrs", label: "of expert video" },
      { value: "13", label: "downstream datasets, new SOTA" },
    ],
  },

  /* ---------------------------------------------------------- */
  "quilt-llava": {
    title: "Quilt-LLaVA: Visual Instruction Tuning from Localized Narratives in Histopathology Videos",
    authors: `Mehmet Saygin Seyfioglu, <b>${ME}</b>, Fatemeh Ghezloo, Ranjay Krishna, Linda Shapiro`,
    venue: "CVPR 2024", year: 2024,
    rung: "grounding",
    tagline: "A patch caption tells you what. Diagnosis needs where — across a gigapixel slide. The narrator's cursor already points to it.",
    teaser: "images/quiltllava.gif",
    links: { arxiv: "https://arxiv.org/abs/2312.04746", pdf: "https://arxiv.org/pdf/2312.04746", web: "https://quilt-llava.github.io/" },
    embed: "https://arxiv.org/pdf/2312.04746",
    beats: [
      { kind: "problem", heading: "Captions without coordinates",
        text: "Existing pathology instruction data describes isolated patches with no spatial grounding and no whole-slide context — so models can't reason across the gigapixel slide the way diagnosis actually requires." },
      { kind: "insight", heading: "Follow the cursor",
        text: "When a narrator says “notice the atypical mitosis here,” their mouse is on it. By automatically tracking cursor position, each spoken concept is tied to the exact region being pointed at — free spatial supervision.",
        figure: "images/quilt_llava2.png",
        caption: "Cursor traces convert narration into spatially grounded instructions." },
      { kind: "method", heading: "Quilt-Instruct",
        text: "From these grounded traces we build Quilt-Instruct: 107K grounded question–answer pairs that force contextual reasoning — extracting a diagnosis plus its supporting evidence across the whole slide — and train Quilt-LLaVA to reason beyond a single patch." },
      { kind: "result", heading: "Reasoning that generalizes",
        text: "Quilt-LLaVA beats the prior state of the art by over 10% on relative GPT-4 evaluation and by 4% (open-set) and 9% (closed-set) on visual question answering, evaluated on 1,283 human-generated QAs over 985 images." },
      { kind: "why", heading: "From seeing to locating",
        text: "Grounding links language to image regions — the bridge from raw perception to diagnostic reasoning, and the same 'where is it' capability a system needs before it can navigate or act." },
    ],
    stats: [
      { value: "107K", label: "grounded QA pairs" },
      { value: ">10%", label: "over SOTA (GPT-4 eval)" },
      { value: "985", label: "images in human eval" },
    ],
  },

  /* ---------------------------------------------------------- */
  "mednarratives": {
    title: "MedicalNarratives: Connecting Medical Vision and Language with Localized Narratives",
    authors: `<b>${ME}</b>, Kevin Zhang, Mehmet Saygin Seyfioglu, Fatemeh Ghezloo, Ranjay Krishna, Linda Shapiro`,
    venue: "NeurIPS 2025", year: 2025,
    rung: "grounding",
    tagline: "The cursor trick worked for pathology. Does the 'think-aloud' teaching signal generalize across all of medical imaging?",
    teaser: "images/mednarratives.gif",
    links: { arxiv: "https://arxiv.org/abs/2501.04184", pdf: "https://arxiv.org/pdf/2501.04184", web: "https://medical-narratives.github.io/" },
    embed: "https://arxiv.org/pdf/2501.04184",
    beats: [
      { kind: "problem", heading: "Scale, but for every modality",
        text: "Medical models are data-hungry, yet unlike natural images, medical imaging can't reach comparable scale — least of all with dense spatial grounding, and least of all across many organ systems at once." },
      { kind: "insight", heading: "Instructors think out loud",
        text: "Across radiology, dermatology, ophthalmology and more, instructors teach by moving the mouse while they narrate. Those movements create spatial–temporal links between speech and image regions — a grounding signal that exists in every domain, not just pathology." },
      { kind: "method", heading: "Traces and boxes at scale",
        text: "We curate mouse traces and bounding boxes tying text to regions from 118K videos with temporally aligned narration across 12 medical domains, then train GenMedClip with a CLIP-like objective." },
      { kind: "result", heading: "4.7M pairs, 1M grounded",
        text: "MedicalNarratives contains 4.7M medical image–text pairs, of which 1M carry dense spatial annotations (traces and bounding boxes). GenMedClip outperforms prior state of the art across all 12 domains on a new benchmark." },
      { kind: "why", heading: "Grounding goes general",
        text: "This scales the grounded-perception foundation from a single organ system to twelve — the cross-modality breadth a general medical intelligence needs before it can specialize into reasoning and action." },
    ],
    stats: [
      { value: "4.7M", label: "image–text pairs" },
      { value: "1M", label: "densely grounded" },
      { value: "12", label: "medical domains" },
    ],
  },

  /* ---------------------------------------------------------- */
  "medblink": {
    title: "MedBLINK: Probing Basic Perception in Multimodal Language Models for Medicine",
    authors: `Mahtab Bigverdi, <b>${ME}</b>, Kevin Zhang, Hyewon Jeong, Mingyu Lu, Sungjae Cho, Linda Shapiro, Ranjay Krishna`,
    venue: "CVAMD Workshop, ICCV 2025", year: 2025,
    rung: "perception",
    tagline: "Before we trust a model to diagnose, can it even tell which way the scan is oriented? Often, no.",
    teaser: "images/medblink_teaser.png",
    links: { arxiv: "https://arxiv.org/abs/2508.02951", pdf: "https://arxiv.org/pdf/2508.02951" },
    embed: "https://arxiv.org/pdf/2508.02951",
    beats: [
      { kind: "problem", heading: "The perception no one checks",
        text: "Medical multimodal LLMs are pitched for diagnostic reasoning — but no one had systematically tested whether they pass the trivially easy perceptual checks clinicians treat as table stakes: image orientation, whether a CT is contrast-enhanced, and more." },
      { kind: "insight", heading: "Isolate perception, make failure unambiguous",
        text: "Rather than another diagnosis benchmark, MedBLINK isolates basic perception into eight clinically meaningful tasks across modalities and anatomy, framed as multiple-choice so a wrong answer is unambiguous.",
        figure: "images/eval_example_1_.jpg",
        caption: "Tasks a clinician finds trivial — orientation, contrast, modality." },
      { kind: "method", heading: "19 models vs. humans",
        text: "1,429 multiple-choice questions over 1,605 images, evaluating 19 state-of-the-art models — both general-purpose (GPT-4o, Claude 3.5 Sonnet) and medical (Med-Flamingo, LLaVA-Med, RadFM) — against human annotators." },
      { kind: "result", heading: "96% vs. 65%",
        text: "Humans reach 96.4% accuracy. The best model manages 65%. Current medical MLMs routinely fail perceptual checks that are near-trivial for people." },
      { kind: "why", heading: "The foundation is cracked",
        text: "Nothing above perception can be trusted until perception is solid. MedBLINK is the honest diagnostic that says: fix the floor before building the tower." },
    ],
    stats: [
      { value: "96.4%", label: "human accuracy" },
      { value: "65%", label: "best model" },
      { value: "19", label: "models probed" },
    ],
  },

  /* ---------------------------------------------------------- */
  "pathfinder": {
    title: "PathFinder: A Multi-Modal Multi-Agent System for Medical Diagnostic Decision-Making in Histopathology",
    authors: `<b>${ME}</b>, Fatemeh Ghezloo, Mehmet Saygin Seyfioglu, Rustin Soraki, Beibin Li, Tejoram Vivekanandan, Joann G. Elmore, Ranjay Krishna, Linda Shapiro`,
    venue: "ICCV 2025", year: 2025,
    rung: "reasoning",
    tagline: "A pathologist doesn't classify a slide in one glance — they triage, navigate, describe, and decide. What if the AI did too?",
    teaser: "images/pathfinder.gif",
    links: { arxiv: "https://arxiv.org/abs/2502.08916", pdf: "https://arxiv.org/pdf/2502.08916" },
    embed: "https://arxiv.org/pdf/2502.08916",
    beats: [
      { kind: "problem", heading: "One glance isn't diagnosis",
        text: "Diagnosing a gigapixel whole-slide image demands the iterative, multi-scale, holistic navigation expert pathologists use. Multiple-instance-learning and transformer models can't replicate it — which limits real-world adoption." },
      { kind: "insight", heading: "Emulate the workflow, not the label",
        text: "PathFinder decomposes diagnosis into four collaborating agents that mirror how a pathologist actually works — triage, navigate, describe, decide — producing not just a class but an explanation." },
      { kind: "method", heading: "Four agents, one decision",
        text: "A Triage Agent flags benign vs. risky; for risky slides, Navigation and Description Agents iteratively focus on significant regions, yielding importance maps and patch-level descriptions; a Diagnosis Agent synthesizes everything into a classification with a natural-language rationale." },
      { kind: "result", heading: "First to beat the pathologist",
        text: "On skin melanoma diagnosis, PathFinder exceeds the prior state of the art by 8% and surpasses average pathologist performance by 9% — the first AI system to do so on this task. Pathologists judged the Description Agent's outputs comparable in quality to GPT-4o." },
      { kind: "why", heading: "Perception + grounding, composed into a policy",
        text: "PathFinder turns static capabilities into an explainable, iterative decision policy over the slide — a direct precursor to the closed-loop navigation and action policies (RL, robotics) at the top of the stack." },
    ],
    stats: [
      { value: "+9%", label: "over avg. pathologist" },
      { value: "+8%", label: "over prior SOTA" },
      { value: "4", label: "collaborating agents" },
    ],
  },

  /* ---------------------------------------------------------- */
  "rubrics": {
    title: "When Rubrics Fail: Error Enumeration as Reward in Reference-Free RL Post-Training for Virtual Try-On",
    authors: `<b>${ME}</b>, Mehmet Saygin Seyfioglu, Ranjay Krishna, Karim Bouyarmane`,
    venue: "ECCV 2026", award: "Spotlight", year: 2026,
    rung: "rl",
    tagline: "How do you reward a generative model when there is no single right answer? Stop scoring against an ideal — start counting what's wrong.",
    teaser: "images/iec.png",
    links: { arxiv: "https://arxiv.org/abs/2603.05659", pdf: "https://arxiv.org/pdf/2603.05659" },
    embed: "https://arxiv.org/pdf/2603.05659",
    beats: [
      { kind: "problem", heading: "No reference, no rubric",
        text: "RL post-training for generative visual tasks usually needs a reference answer or a rubric scoring against an 'ideal' output. For tasks like virtual try-on — and, tellingly, most of medicine — no single ground-truth reference exists." },
      { kind: "insight", heading: "Invert the reward",
        text: "Instead of scoring how close an output is to an ideal, Implicit Error Counting (IEC) enumerates what the model got wrong and applies severity-weighted penalties across task-relevant axes. You don't need the perfect answer to recognize a mistake." },
      { kind: "method", heading: "Making error-counting stable",
        text: "Naive error enumeration is too noisy to serve as a reward, so IEC adds implicit scoring and group calibration for reliability. A Cascaded Error Counting (CEC) metric and a new benchmark, Mismatch-DressCode, support evaluation." },
      { kind: "result", heading: "Aligned with human judgment",
        text: "CEC aligns with human preferences at 60% top-1 versus 30% for alternatives. On MDressBench, IEC matches or exceeds six baselines on 6 of 8 perceptual metrics on VITON-HD and DressCode." },
      { kind: "why", heading: "The reward machinery for open-ended domains",
        text: "Reference-free, severity-weighted error rewards are exactly what RL post-training needs where 'the one correct answer' rarely exists but enumerable, weighted errors do — the self-improvement engine for medicine and surgery." },
    ],
    stats: [
      { value: "60%", label: "human-pref top-1 (vs 30%)" },
      { value: "6/8", label: "perceptual metrics matched/beaten" },
      { value: "0", label: "reference answers required" },
    ],
  },

  /* ---------------------------------------------------------- */
  "svg2": {
    title: "Synthetic Visual Genome 2: Extracting Large-Scale Spatio-Temporal Scene Graphs from Videos",
    authors: `Ziqi Gao, Jieyu Zhang, <b>${ME}</b>, Jae Sung Park, Tario G. You, Daniel Ogbu, Chenhao Zheng, Weikai Huang, Yinuo Yang, Winson Han, Quan Kong, Rajat Saini, Ranjay Krishna`,
    venue: "ECCV 2026", year: 2026,
    rung: "embodiment",
    tagline: "To act in the world, a model must know who is doing what to what, where, and when. That structure has to be learned at scale.",
    teaser: "images/traser.png",
    links: { arxiv: "https://arxiv.org/abs/2602.23543", web: "https://uwgzq.github.io/papers/SVG2/" },
    embed: "https://arxiv.org/pdf/2602.23543",
    beats: [
      { kind: "problem", heading: "Structure at the wrong scale",
        text: "Video scene-graph datasets are far too small and sparsely annotated to teach dense, temporally grounded structure — the who/what/where/when that any embodied policy must track." },
      { kind: "insight", heading: "Synthesize the graph",
        text: "An automated pipeline chains panoptic segmentation, trajectory tracking, semantic parsing, and relation inference to synthesize panoptic spatio-temporal scene graphs at scale, then validates them with human verification." },
      { kind: "method", heading: "TRaSER binds entities through time",
        text: "On top of the dataset we train TRaSER, a scene-graph model with trajectory-aligned token mechanisms that keep entity identity stable across frames — so a relation isn't just detected, it's tracked." },
      { kind: "result", heading: "Dense, verified, and it helps",
        text: "636K+ videos; 6.6M objects, 52.0M attributes, 6.7M relations, with human-verified accuracy of 93.8% (objects), 88.3% (attributes), 85.4% (relations). TRaSER gains +15–20% relation detection and +30–40% object prediction over baselines, and +1.5–4.6% absolute on video QA." },
      { kind: "why", heading: "The substrate for surgical video",
        text: "Spatio-temporal scene graphs are the grounding layer between perception and action — the same machinery needed to track instruments, anatomy, and actions through surgical video, a prerequisite for RL policies and surgical robotics." },
    ],
    stats: [
      { value: "636K+", label: "videos" },
      { value: "52M", label: "attributes" },
      { value: "+15–20%", label: "relation detection" },
    ],
  },

  /* ---------------------------------------------------------- */
  "mscamil": {
    title: "Multi-Scale Cross-Attention Multiple Instance Learning for Classification of Colorectal Polyps",
    authors: `<b>${ME}</b>, Yongjun Liu, Kareem Hosny, Jonathan Henriksen, Robert W. Ricciotti, Meenal Rawlani, Paul E. Swanson, Patrick C. Mathias, Noah Hoffman, Geoffrey S. Baird, Luis F. Gonzalez-Cuyar, Linda G. Shapiro, Deepti M. Reddi`,
    venue: "Cancer Informatics (SAGE), 2026", year: 2026,
    rung: "perception",
    tagline: "Pathologists read slides across magnifications at once. A model that attends across scales can triage polyps the same way.",
    teaser: "images/multiscale_.png",
    links: { journal: "https://journals.sagepub.com/home/cix", pdf: "https://arxiv.org/pdf/2209.01534" },
    embed: "https://arxiv.org/pdf/2209.01534",
    beats: [
      { kind: "problem", heading: "Gigapixel triage",
        text: "Whole-slide colorectal polyp classification requires reasoning over enormous images where the diagnostic signal lives at multiple magnifications simultaneously." },
      { kind: "insight", heading: "Cross-attention across scales",
        text: "A multi-scale cross-attention multiple-instance-learning network (MsCAMIL) lets features at different magnifications inform one another, mirroring how a pathologist zooms in and out while forming a judgment." },
      { kind: "why", heading: "Clinical-grade perception",
        text: "Validated with practicing pathologists and presented at USCAP, this is the applied, deployment-facing edge of the perception layer — real slides, real clinical workflows." },
    ],
    stats: [],
  },

  /* ---------------------------------------------------------- */
  "mmae": {
    title: "Multi-Modal Masked Autoencoders Learn Compositional Histopathological Representations",
    authors: `<b>${ME}</b>, Mehmet Saygin Seyfioglu, Linda Shapiro`,
    venue: "ML4H 2022 (Extended Abstract)", year: 2022,
    rung: "perception",
    tagline: "An early bet: reconstruct masked image and text jointly, and the representation learns the compositional structure of tissue.",
    teaser: "images/mmae2.png",
    links: { pdf: "https://arxiv.org/pdf/2209.01534" },
    embed: "https://arxiv.org/pdf/2209.01534",
    beats: [
      { kind: "insight", heading: "Mask both modalities",
        text: "Multi-modal masked autoencoding over paired histopathology image and text learns representations that capture compositional structure — a self-supervised precursor to the contrastive foundations that followed in Quilt." },
      { kind: "why", heading: "Where the thread starts",
        text: "This early work planted the question that Quilt-1M would answer at scale: how do we learn medical visual–linguistic structure without hand labels?" },
    ],
    stats: [],
  },

  /* ---------------------------------------------------------- */
  "eeg-dg": {
    title: "Supervised Domain Generalization for Integration of Disparate Scalp EEG Datasets for Automatic Epileptic Seizure Detection",
    authors: `K. P. Ayodele, <b>${ME}</b>, M. A. Komolafe, P. Ogunbona`,
    venue: "Computers in Biology and Medicine, 2020", year: 2020,
    rung: "origins",
    tagline: "Before pixels, there were signals. The first question was already the recurring one: can we combine disparate datasets without the distribution shift breaking the model?",
    teaser: "images/cbm2020.png",
    links: { journal: "https://doi.org/10.1016/j.compbiomed.2020.103757" },
    embed: "",
    beats: [
      { kind: "problem", heading: "Datasets that won't combine",
        text: "Automatic seizure detection was starved of data because disparate scalp-EEG datasets, recorded under different conditions, couldn't simply be pooled — distribution shift degraded any model trained on the union." },
      { kind: "insight", heading: "Generalize across the shift",
        text: "By minimizing the distributional shift between datasets, disparate EEG corpora can be integrated to increase trainable dataset size, achieving results comparable to large-scale validation studies." },
      { kind: "why", heading: "The origin of the obsession",
        text: "Undergraduate work with Prof. Kayode Ayodele. The core theme — combine scarce, heterogeneous medical data to unlock learning — is the same one that would later drive Quilt-1M and MedicalNarratives." },
    ],
    stats: [],
  },

};
