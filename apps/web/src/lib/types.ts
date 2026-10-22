// Types mirroring the FastAPI schemas in apps/api/app/schemas.

export type Task = "language" | "entities" | "grammar";
export type PolicyName = "tuned" | "reference";
export type LanguageSetting = "auto" | string;

export interface Language {
  code: string; // "und" means inconclusive
  name: string;
  alt_name: string | null;
}

export interface LanguageDetail extends Language {
  kurdish: boolean;
  script: string;
  direction: "rtl" | "ltr" | "auto";
}

export interface ClassScore {
  code: string;
  name: string;
  score: number;
}

export interface ModelRef {
  id: string;
  name: string;
  identifier: string;
  version: string;
}

export interface PolicyInfo {
  name: PolicyName;
  calibrated: boolean;
  confidence_threshold: number;
  margin_threshold: number;
  minimum_letter_count: number;
}

export interface Notice {
  code: string;
  severity: "info" | "warning";
  message: string;
}

export interface InputStats {
  characters: number;
  letters: number;
  words: number;
  script: string;
}

export interface DetectResponse {
  language: Language;
  is_reliable: boolean;
  abstained: boolean;
  reason: string | null;
  reason_message: string | null;
  confidence: number;
  margin: number;
  scores: ClassScore[];
  model: ModelRef;
  policy: PolicyInfo;
  score_semantics: string;
  warnings: Notice[];
  input: InputStats;
  elapsed_ms: number;
}

export interface TaskModelRef {
  id: string;
  name: string;
  version: string | null;
}

/** Offsets are Unicode code point offsets into the submitted text. */
export interface Entity {
  text: string;
  label: string;
  start: number;
  end: number;
}

export interface EntitiesResult {
  model: TaskModelRef;
  labels: string[];
  entities: Entity[];
}

export interface Token {
  index: number;
  text: string;
  start: number;
  end: number;
  lemma: string | null;
  pos: string | null;
  tag: string | null;
  morph: Record<string, string>;
  dep: string | null;
  head: number;
}

export interface GrammarResult {
  model: TaskModelRef;
  sentences: { start: number; end: number }[];
  tokens: Token[];
}

export interface AnalyzeResponse {
  task: Task;
  language: Language & { mode: "auto" | "manual" };
  detection: DetectResponse | null;
  entities: EntitiesResult | null;
  grammar: GrammarResult | null;
}

export interface InternalTestMetrics {
  scope: "internal";
  dataset: string;
  records: number | null;
  macro_f1: number;
  accuracy: number | null;
}

export interface ExternalBenchmark {
  name: string;
  description: string;
  records: number | null;
  expected_label: string;
  recall: number;
  correct: number | null;
}

export interface ModelInfo {
  id: string;
  name: string;
  full_name: string;
  task: Task;
  description: string;
  languages: string[];
  status: "ready" | "unavailable";
  default: boolean;
  unavailable_reasons: string[];
  identifier: string | null;
  version: string | null;
  experiment: string | null;
  framework: string | null;
  size_bytes: number | null;
  temperature: number | null;
  policies: PolicyInfo[];
  artifact_sha256: string | null;
  internal_test: InternalTestMetrics | null;
  external_benchmarks: ExternalBenchmark[];
}

export interface TaskInfo {
  id: Task;
  name: string;
  available: boolean;
  message: string | null;
  languages: string[];
}

export interface ModelsResponse {
  models: ModelInfo[];
  tasks: TaskInfo[];
  languages: LanguageDetail[];
  limits: { max_text_chars: number };
  default_model: string | null;
  documentation_url: string;
}

export interface AnalyzeRequest {
  text: string;
  task: Task;
  language: LanguageSetting;
  model?: string;
  policy?: PolicyName;
}

export interface DetectRequest {
  text: string;
  model?: string;
  policy?: PolicyName;
  top_k?: number;
}
