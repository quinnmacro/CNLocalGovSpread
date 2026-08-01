import { ScenariosContent } from "./scenarios-content";

const pageInfo = {
  title: "情景分析",
  subtitle: "Scenarios — 蒙特卡洛模拟与压力测试",
};

export const metadata = { title: pageInfo.title };

export default function Page() {
  return <ScenariosContent />;
}
