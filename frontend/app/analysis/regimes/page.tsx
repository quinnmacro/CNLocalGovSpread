import { RegimesContent } from "./regimes-content";

const pageInfo = {
  title: "市场状态",
  subtitle: "Regimes — HMM + MarketGauge 状态识别",
};

export const metadata = { title: `${pageInfo.title} | QuinnMacro` };

export default function Page() {
  return <RegimesContent />;
}
