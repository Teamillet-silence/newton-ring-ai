const GREEK: Record<string, string> = {
  alpha: "阿尔法",
  beta: "贝塔",
  gamma: "伽马",
  delta: "德尔塔",
  epsilon: "伊普西龙",
  zeta: "泽塔",
  eta: "伊塔",
  theta: "西塔",
  iota: "约塔",
  kappa: "卡帕",
  lambda: "兰布达",
  mu: "缪",
  nu: "纽",
  xi: "克西",
  omicron: "奥米克戎",
  pi: "派",
  rho: "柔",
  sigma: "西格马",
  tau: "套",
  upsilon: "宇普西龙",
  phi: "斐",
  chi: "卡西",
  psi: "普西",
  omega: "欧米伽",
};

const FUNCTIONS: Record<string, string> = {
  sin: "正弦",
  cos: "余弦",
  tan: "正切",
  cot: "余切",
  log: "对数",
  ln: "自然对数",
  lg: "常用对数",
  lim: "极限",
  max: "最大值",
  min: "最小值",
};

const SYMBOLS: Record<string, string> = {
  "infty": "无穷大",
  "rightarrow": "趋向于",
  "Rightarrow": "推出",
  "leftarrow": "来自于",
  "Leftarrow": "由推出",
  "approx": "约等于",
  "neq": "不等于",
  "equiv": "恒等于",
  "propto": "正比例于",
  "sim": "相似于",
  "cong": "全等于",
  "perp": "垂直于",
  "parallel": "平行于",
  "angle": "角",
  "triangle": "三角形",
  "cdot": "点乘",
  "times": "乘以",
  "div": "除以",
  "pm": "正负",
  "mp": "负正",
  "le": "小于等于",
  "ge": "大于等于",
  "ll": "远小于",
  "gg": "远大于",
  "ne": "不等于",
  "subset": "包含于",
  "supset": "包含",
  "cup": "并集",
  "cap": "交集",
  "emptyset": "空集",
  "nabla": "梯度",
  "partial": "偏微分",
  "int": "积分",
  "oint": "环路积分",
  "iint": "二重积分",
  "iiint": "三重积分",
  "sum": "求和",
  "prod": "求积",
  "coprod": "余积",
};

const ACCENTS: Record<string, string> = {
  bar: "拔",
  vec: "向量",
  dot: "点",
  hat: "帽",
  tilde: "波浪",
};

function convertGreek(text: string): string {
  return text.replace(/\\([a-zA-Z]+)/g, (_, name) => {
    const lower = name.toLowerCase();
    if (GREEK[lower]) return GREEK[lower];
    if (FUNCTIONS[lower]) return FUNCTIONS[lower];
    if (SYMBOLS[name]) return SYMBOLS[name];
    if (ACCENTS[lower]) return ACCENTS[lower];
    return "";
  });
}

function convertSqrt(match: string, content: string): string {
  const inner = convertLatexToSpeech(content);
  return inner ? `根号${inner}` : "根号";
}

function convertFrac(match: string, num: string, den: string): string {
  const n = convertLatexToSpeech(num);
  const d = convertLatexToSpeech(den);
  return `${d}分之${n}`;
}

function cleanSubSuper(text: string): string {
  return text
    .replace(/\{([^}]*)\}/g, "$1")
    .replace(/\\( )/g, " ");
}

export function convertLatexToSpeech(latex: string): string {
  let result = latex;

  result = result.replace(/\\frac\{([^}]*)\}\{([^}]*)\}/g, convertFrac);
  result = result.replace(/\\sqrt(\[([^\]]*)\])?\{([^}]*)\}/g, (_, __, root, content) => {
    const inner = convertLatexToSpeech(content);
    return root ? `${root}次根号${inner}` : `根号${inner}`;
  });

  result = convertGreek(result);

  result = result
    .replace(/\^{([^}]*)}/g, (_, exp) => {
      const e = convertLatexToSpeech(exp);
      if (/^\d+$/.test(exp)) return `的${exp}次方`;
      return `的${e}次方`;
    })
    .replace(/\^([a-zA-Z0-9])/g, (_, c) => {
      if (/^\d$/.test(c)) return `的${c}次方`;
      return `的${c}次方`;
    })
    .replace(/_{([^}]*)}/g, (_, sub) => `下标${convertLatexToSpeech(sub)}`)
    .replace(/_([a-zA-Z0-9])/g, (_, c) => `下标${c}`);

  result = result
    .replace(/\\operatorname\{([^}]*)\}/g, "$1")
    .replace(/\\text\{([^}]*)\}/g, "$1")
    .replace(/\\mathrm\{([^}]*)\}/g, "$1")
    .replace(/\\mathbf\{([^}]*)\}/g, "$1")
    .replace(/\\mathit\{([^}]*)\}/g, "$1");

  result = result
    .replace(/\\left/g, "")
    .replace(/\\right/g, "")
    .replace(/\\big/g, "")
    .replace(/\\bigg/g, "")
    .replace(/\\bigl/g, "")
    .replace(/\\bigr/g, "")
    .replace(/\\biggl/g, "")
    .replace(/\\biggr/g, "");

  result = result
    .replace(/[{}]/g, "")
    .replace(/\\,/g, " ")
    .replace(/\\;/g, " ")
    .replace(/\\:/g, " ")
    .replace(/\\!/g, "")
    .replace(/\\quad/g, " ")
    .replace(/\\qquad/g, "  ")
    .replace(/\\s/g, " ");

  result = result.replace(/\(/g, "").replace(/\)/g, "").replace(/\[/g, "").replace(/\]/g, "");

  result = result.replace(/\s+/g, " ").trim();

  return result;
}

export function cleanLatexForSpeech(text: string): string {
  return text
    .replace(/\$\$([\s\S]*?)\$\$/g, (_, math) => convertLatexToSpeech(math))
    .replace(/\$([^$]*?)\$/g, (_, math) => convertLatexToSpeech(math))
    .replace(/\\\(([\s\S]*?)\\\)/g, (_, math) => convertLatexToSpeech(math))
    .replace(/\\\[([\s\S]*?)\\\]/g, (_, math) => convertLatexToSpeech(math))
    .replace(/#{1,6}\s/g, "")
    .replace(/[*_~`]/g, "")
    .replace(/\n{3,}/g, "\n")
    .trim();
}
