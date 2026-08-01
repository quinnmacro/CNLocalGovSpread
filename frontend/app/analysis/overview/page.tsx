import { OverviewContent } from "./overview-content";

const pageInfo = {
  title: "利差全景",
  subtitle: "Overview — 趋势、分布、期限结构、分布特征",
};

export const metadata = { title: `${pageInfo.title} | QuinnMacro` };

export default function Page() {
  return <OverviewContent />;
}
