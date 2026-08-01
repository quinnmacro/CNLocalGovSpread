import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Providers } from "@/components/providers";
import { Navbar } from "@/components/layout/navbar";
import { Footer } from "@/components/layout/footer";
import "./globals.css";

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "中国地方政府债信用利差 — 量化分析平台 | QuinnMacro",
    template: "%s | QuinnMacro",
  },
  description:
    "基于 GARCH/EVT/HMM 的中国地方政府债信用利差建模、风险度量与状态监控系统。2018-2026 年日频数据，7 种波动率模型，极值理论尾部分析，隐马尔可夫状态识别。",
  keywords: [
    "地方政府债",
    "信用利差",
    "GARCH",
    "极值理论",
    "HMM",
    "量化研究",
    "固收",
    "QuinnMacro",
  ],
  openGraph: {
    title: "中国地方政府债信用利差 — 量化分析平台",
    description: "GARCH/EVT/HMM 驱动的固收量化研究平台",
    type: "website",
    locale: "zh_CN",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="zh"
      className={`dark ${inter.variable} ${jetbrainsMono.variable}`}
      suppressHydrationWarning
    >
      <body className="min-h-screen flex flex-col antialiased">
        <Providers>
          <TooltipProvider delay={200}>
            <Navbar />
            <main className="flex-1">{children}</main>
            <Footer />
          </TooltipProvider>
        </Providers>
      </body>
    </html>
  );
}
