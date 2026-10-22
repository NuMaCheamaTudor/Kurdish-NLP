// Sample inputs from the Dataset V1 *train* split (never dev, test or the UD
// benchmark). Source: Tatoeba (tatoeba.org), CC BY 2.0 FR.

export interface Example {
  id: string;
  label: string;
  text: string;
}

export const EXAMPLE_SOURCE = "Tatoeba · CC BY 2.0 FR";

export const EXAMPLES: Example[] = [
  { id: "ckb", label: "Sorani", text: "زۆر جار ناوی ئەو کەسانەم لەبیر دەچێتەوە کە چاوم پێیان دەکەوێت." },
  { id: "kmr", label: "Kurmanji", text: "Koma dîlanê govendên gelêrî yên cur bi cur gerandin." },
  { id: "sdh", label: "Southern Kurdish", text: "مە تووام بچم ئەڕا ئەورا و بامەو ئەڕا ئێرا." },
];
