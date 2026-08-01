import type { Metadata } from "next";
import { RiskContent } from "./risk-content";

const pageInfo = { title: "风险度量", subtitle: "Risk — VaR / ES / EVT 尾部分析" };
export const metadata: Metadata = { title: pageInfo.title };

export default function Page() {
  return <RiskContent />;
}
