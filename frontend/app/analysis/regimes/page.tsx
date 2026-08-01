import { RegimesContent } from "./regimes-content";

const pageInfo = {
  title: "市场状态",
  subtitle: "Regimes — HMM + STS + Bayesian + CPD 多维状态识别",
};

export const metadata = { title: pageInfo.title };

export default function Page() {
  return <RegimesContent />;
}
