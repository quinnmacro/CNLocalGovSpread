"""
报告生成模块 - PDF/Excel/HTML报告生成

功能:
1. 分析报告生成
2. 多格式导出 (PDF, Excel, HTML)
3. 历史报告管理
"""

import os
import io
import json
from datetime import datetime
import pandas as pd
import numpy as np


# ============================================================================
# P0修复: 免责声明
# ============================================================================

DISCLAIMER = """
⚠️ 重要声明 (Disclaimer)

本报告仅供学术研究和教育目的，不构成任何投资建议。

1. 模型局限性：所有计量经济学模型都是对现实的简化，实际市场行为可能偏离模型预测。
2. 历史不代表未来：基于历史数据的统计特征不保证在未来延续。
3. 风险提示：投资有风险，决策需谨慎。请在专业人士指导下做出投资决策。
4. 免责：报告作者不对因使用本报告内容而导致的任何损失承担责任。

本报告基于CNLocalGovSpread框架生成，版本: 3.0.0
"""


# ============================================================================
# 报告生成器
# ============================================================================

class ReportGenerator:
    """报告生成器类"""

    def __init__(self, output_dir='reports'):
        """
        初始化报告生成器

        参数:
            output_dir: 报告输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.report_history_file = os.path.join(output_dir, 'history.json')
        self._load_history()

    def _load_history(self):
        """加载报告历史"""
        if os.path.exists(self.report_history_file):
            with open(self.report_history_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = []

    def _save_history(self):
        """保存报告历史"""
        with open(self.report_history_file, 'w') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def generate_report(self, clean_data, returns, kalman, vol_modeler, evt,
                       title="地方债利差分析报告", format="PDF", sections=None):
        """
        生成分析报告

        参数:
            clean_data: 清洗后的数据
            returns: 收益率序列
            kalman: 卡尔曼滤波器
            vol_modeler: 波动率建模器
            evt: EVT分析器
            title: 报告标题
            format: 报告格式 (PDF, Excel, HTML)
            sections: 包含的章节列表

        返回:
            str: 报告文件路径
        """
        if sections is None:
            sections = ['数据概览', '信号分析', '波动率分析', '风险分析', '交易建议']

        # 生成报告内容
        report_data = self._prepare_report_data(
            clean_data, returns, kalman, vol_modeler, evt, sections
        )

        # 根据格式生成报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if format.upper() == 'PDF':
            path = self._generate_pdf(report_data, title, timestamp)
        elif format.upper() == 'EXCEL':
            path = self._generate_excel(report_data, title, timestamp)
        elif format.upper() == 'HTML':
            path = self._generate_html(report_data, title, timestamp)
        else:
            raise ValueError(f"不支持的格式: {format}")

        # 记录历史
        self._add_to_history(title, format, path, sections)

        return path

    def _prepare_report_data(self, clean_data, returns, kalman, vol_modeler, evt, sections):
        """准备报告数据"""
        data = {}

        # 数据概览
        if '数据概览' in sections:
            spread = clean_data['spread']
            data['overview'] = {
                '数据范围': f"{clean_data.index[0].strftime('%Y-%m-%d')} 至 {clean_data.index[-1].strftime('%Y-%m-%d')}",
                '交易日数': len(clean_data),
                '当前利差': f"{spread.iloc[-1]:.4f}",
                '历史均值': f"{spread.mean():.4f}",
                '历史标准差': f"{spread.std():.4f}",
                '最小值': f"{spread.min():.4f}",
                '最大值': f"{spread.max():.4f}",
                '偏度': f"{spread.skew():.4f}",
                '峰度': f"{spread.kurt():.4f}"
            }

        # 信号分析
        if '信号分析' in sections and kalman is not None:
            smoothed = kalman.smoothed_state
            deviation = kalman.get_signal_deviation()
            current_dev = deviation.iloc[-1]

            signal = '中性'
            if current_dev > 1.5:
                signal = '做空（利差高估）'
            elif current_dev < -1.5:
                signal = '做多（利差低估）'

            data['signal'] = {
                '当前利差': f"{spread.iloc[-1]:.4f}",
                '趋势水平': f"{smoothed.iloc[-1]:.4f}",
                '偏离度': f"{current_dev:.4f}σ",
                '交易信号': signal
            }

        # 波动率分析
        if '波动率分析' in sections and vol_modeler is not None:
            winner = vol_modeler.run_tournament()
            winner_vol = vol_modeler.get_conditional_volatility(winner)

            data['volatility'] = {
                '获胜模型': winner,
                'AIC': f"{vol_modeler.ic_scores[winner]['AIC']:.2f}",
                'BIC': f"{vol_modeler.ic_scores[winner]['BIC']:.2f}",
                '当前波动率': f"{winner_vol.iloc[-1]:.6f}",
                '平均波动率': f"{winner_vol.mean():.6f}"
            }

        # 风险分析
        if '风险分析' in sections and evt is not None:
            var = evt.calculate_var() if evt.var is None else evt.var
            es = evt.calculate_es() if evt.es is None else evt.es
            tail_idx = evt.get_tail_index()

            data['risk'] = {
                '99% VaR': f"{var:.6f}" if var else "N/A",
                '99% ES': f"{es:.6f}" if es else "N/A",
                '尾部指数': f"{tail_idx:.4f}" if tail_idx else "N/A",
                'GPD形状参数': f"{evt.gpd_params['shape']:.4f}" if evt.gpd_params else "N/A",
                'GPD尺度参数': f"{evt.gpd_params['scale']:.4f}" if evt.gpd_params else "N/A"
            }

        # 交易建议
        if '交易建议' in sections:
            data['recommendation'] = self._generate_recommendation(
                clean_data, kalman, evt, vol_modeler
            )

        return data

    def _generate_recommendation(self, clean_data, kalman, evt, vol_modeler):
        """生成交易建议"""
        spread = clean_data['spread']
        current = spread.iloc[-1]
        mean = spread.mean()

        # 综合判断
        signals = []

        # 偏离度信号
        if kalman is not None:
            deviation = kalman.get_signal_deviation().iloc[-1]
            if deviation > 1.5:
                signals.append(('做空', deviation, '利差高估'))
            elif deviation < -1.5:
                signals.append(('做多', deviation, '利差低估'))
            else:
                signals.append(('中性', deviation, '利差正常'))

        # 趋势信号
        recent_mean = spread.iloc[-20:].mean()
        if recent_mean > mean:
            signals.append(('观望', 0, '上升趋势'))
        elif recent_mean < mean:
            signals.append(('观望', 0, '下降趋势'))

        # 综合建议
        recommendation = {
            '当前利差': f"{current:.4f}",
            '历史均值': f"{mean:.4f}",
            '建议': signals[0][0] if signals else '中性',
            '理由': signals[0][2] if signals else '数据正常',
            '止损建议': f"{current + (evt.var if evt and evt.var else spread.std()):.4f}"
        }

        return recommendation

    def _generate_pdf(self, data, title, timestamp):
        """生成PDF报告"""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib import colors
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont

            # 注册中文字体（跨平台支持）
            # P0修复: 移除硬编码macOS路径，使用跨平台方案
            chinese_font = 'Helvetica'  # 默认英文字体

            font_paths = [
                # macOS
                '/System/Library/Fonts/PingFang.ttc',
                '/System/Library/Fonts/STHeiti Light.ttc',
                '/Library/Fonts/Arial Unicode.ttf',
                # Windows
                'C:/Windows/Fonts/simhei.ttf',
                'C:/Windows/Fonts/msyh.ttc',
                # Linux
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            ]

            for font_path in font_paths:
                try:
                    if os.path.exists(font_path):
                        pdfmetrics.registerFont(TTFont('SimHei', font_path))
                        chinese_font = 'SimHei'
                        break
                except Exception:
                    continue

            filename = os.path.join(self.output_dir, f"report_{timestamp}.pdf")
            doc = SimpleDocTemplate(filename, pagesize=A4)

            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontName=chinese_font,
                fontSize=18,
                spaceAfter=30
            )
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontName=chinese_font,
                fontSize=14,
                spaceAfter=12
            )

            elements = []

            # 标题
            elements.append(Paragraph(title, title_style))
            elements.append(Paragraph(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            elements.append(Spacer(1, 1*cm))

            # 各章节
            for section, content in data.items():
                elements.append(Paragraph(section, heading_style))

                if isinstance(content, dict):
                    table_data = [[k, v] for k, v in content.items()]
                    table = Table(table_data, colWidths=[6*cm, 8*cm])
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('FONTNAME', (0, 0), (-1, -1), chinese_font),
                        ('FONTSIZE', (0, 0), (-1, -1), 10),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ]))
                    elements.append(table)
                    elements.append(Spacer(1, 0.5*cm))

            doc.build(elements)
            return filename

        except ImportError:
            # 如果没有 reportlab，回退到文本报告
            return self._generate_text(data, title, timestamp, 'txt')

    def _generate_excel(self, data, title, timestamp):
        """生成Excel报告"""
        filename = os.path.join(self.output_dir, f"report_{timestamp}.xlsx")

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 摘要页
            summary_data = []
            for section, content in data.items():
                if isinstance(content, dict):
                    for key, value in content.items():
                        summary_data.append({
                            '章节': section,
                            '指标': key,
                            '值': value
                        })

            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='报告摘要', index=False)

            # 各章节单独页
            for section, content in data.items():
                if isinstance(content, dict):
                    df = pd.DataFrame([content]).T
                    df.columns = ['值']
                    df.index.name = '指标'
                    df.to_excel(writer, sheet_name=section[:31])  # Excel sheet名最长31字符

        return filename

    def _generate_html(self, data, title, timestamp):
        """生成HTML报告"""
        filename = os.path.join(self.output_dir, f"report_{timestamp}.html")

        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1E3A5F;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #667eea;
            margin-top: 30px;
        }}
        .meta {{
            color: #666;
            font-size: 0.9rem;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #667eea;
            color: white;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
            font-size: 0.8rem;
        }}
        .disclaimer {{
            margin-top: 30px;
            padding: 20px;
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            font-size: 0.85rem;
            color: #856404;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p class="meta">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <div class="disclaimer">
            <strong>⚠️ 重要声明</strong><br>
            本报告仅供学术研究和教育目的，不构成任何投资建议。所有模型都是对现实的简化，历史表现不代表未来收益。投资有风险，决策需谨慎。
        </div>
"""

        for section, content in data.items():
            html += f"        <h2>{section}</h2>\n"
            html += "        <table>\n"
            html += "            <tr><th>指标</th><th>值</th></tr>\n"
            if isinstance(content, dict):
                for key, value in content.items():
                    html += f"            <tr><td>{key}</td><td>{value}</td></tr>\n"
            html += "        </table>\n"

        html += f"""
        <div class="footer">
            <p>CNLocalGovSpread v3.0.0 | Author: Quinn Liu</p>
            <p><a href="https://github.com/quinnmacro/CNLocalGovSpread">GitHub</a> | <a href="https://www.linkedin.com/in/liulu-math">LinkedIn</a></p>
        </div>
    </div>
</body>
</html>
"""

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)

        return filename

    def _generate_text(self, data, title, timestamp, ext='txt'):
        """生成文本报告"""
        filename = os.path.join(self.output_dir, f"report_{timestamp}.{ext}")

        lines = [
            "=" * 60,
            title,
            "=" * 60,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]

        for section, content in data.items():
            lines.append(f"\n{section}")
            lines.append("-" * 40)
            if isinstance(content, dict):
                for key, value in content.items():
                    lines.append(f"  {key}: {value}")

        lines.append("\n" + "=" * 60)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return filename

    def _add_to_history(self, title, format, path, sections):
        """添加到历史记录"""
        record = {
            'title': title,
            'format': format,
            'path': path,
            'sections': sections,
            'timestamp': datetime.now().isoformat(),
            'filename': os.path.basename(path)
        }
        self.history.insert(0, record)
        self._save_history()

    def get_history(self):
        """获取报告历史"""
        return self.history

    def delete_report(self, filename):
        """删除报告"""
        path = os.path.join(self.output_dir, filename)
        if os.path.exists(path):
            os.remove(path)
            self.history = [r for r in self.history if r.get('filename') != filename]
            self._save_history()
            return True
        return False


# ============================================================================
# 便捷函数
# ============================================================================

def generate_report(clean_data, returns, kalman, vol_modeler, evt,
                   title="地方债利差分析报告", format="PDF", sections=None):
    """
    生成分析报告的便捷函数

    参数:
        clean_data: 清洗后的数据
        returns: 收益率序列
        kalman: 卡尔曼滤波器
        vol_modeler: 波动率建模器
        evt: EVT分析器
        title: 报告标题
        format: 报告格式
        sections: 包含的章节

    返回:
        str: 报告文件路径
    """
    generator = ReportGenerator()
    return generator.generate_report(
        clean_data, returns, kalman, vol_modeler, evt,
        title=title, format=format, sections=sections
    )


def get_report_history():
    """获取报告历史的便捷函数"""
    generator = ReportGenerator()
    history = generator.get_history()

    if not history:
        return None

    return pd.DataFrame(history)


def generate_quick_report(clean_data, returns, kalman, vol_modeler, evt):
    """生成快速报告（包含所有章节）"""
    return generate_report(
        clean_data, returns, kalman, vol_modeler, evt,
        title="地方债利差快速分析报告",
        format="HTML",
        sections=['数据概览', '信号分析', '波动率分析', '风险分析', '交易建议']
    )
