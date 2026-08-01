import { VolatilityContent } from "./volatility-content";

const pageInfo = {
  title: "波动率建模",
  subtitle: "Volatility — GARCH 族模型对比与诊断",
};

export const metadata = { title: `${pageInfo.title} | QuinnMacro` };

export default function Page() {
  return <VolatilityContent />;
}
