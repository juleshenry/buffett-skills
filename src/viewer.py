from pathlib import Path
from flask import Flask, render_template_string
import glob, json, os
import comparison_scoring

app = Flask(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"
REFERENCES_DIR = ROOT_DIR / "skills" / "buffett" / "references"
PRINCIPLES_DIR = ROOT_DIR / "skills" / "buffett" / "principles"

import re

def camel_to_spaces(s):
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', s)

app.jinja_env.filters['camel_to_spaces'] = camel_to_spaces


TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Buffett Skills Viewer</title>
    <script src="https://cdn.tailwindcss.com?plugins=typography"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    typography: {
                        DEFAULT: {
                            css: {
                                maxWidth: 'none',
                                h1: { fontSize: '2.25rem', fontWeight: '800', marginTop: '2rem', marginBottom: '1rem' },
                                h2: { fontSize: '1.875rem', fontWeight: '700', marginTop: '1.5rem', marginBottom: '0.75rem' },
                                h3: { fontSize: '1.5rem', fontWeight: '600', marginTop: '1.25rem', marginBottom: '0.5rem' },
                                p: { marginTop: '0.75rem', marginBottom: '0.75rem', lineHeight: '1.75' },
                                ul: { listStyleType: 'disc', paddingLeft: '1.5rem', marginTop: '0.5rem', marginBottom: '0.5rem' },
                                ol: { listStyleType: 'decimal', paddingLeft: '1.5rem', marginTop: '0.5rem', marginBottom: '0.5rem' },
                                li: { marginTop: '0.25rem', marginBottom: '0.25rem' },
                                pre: { backgroundColor: '#1f2937', color: '#e5e7eb', padding: '1rem', borderRadius: '0.5rem', overflowX: 'auto' },
                                code: { backgroundColor: '#f3f4f6', padding: '0.2rem 0.4rem', borderRadius: '0.25rem', fontSize: '0.875em' },
                                blockquote: { borderLeftWidth: '4px', borderLeftColor: '#d1d5db', paddingLeft: '1rem', fontStyle: 'italic', color: '#4b5563' }
                            }
                        }
                    }
                }
            }
        }

        const reportsData = {{ reports | tojson }};

        function showHeuristicDetails(ticker, category, heuristic) {
            const detailsContainer = document.getElementById('details-' + ticker);
            const titleEl = document.getElementById('details-title-' + ticker);
            const contentEl = document.getElementById('details-content-' + ticker);
            
            const rawData = reportsData[ticker][category][heuristic];
            
            // Format title
            titleEl.textContent = heuristic.replace(/([A-Z])/g, ' $1').trim();
            
            // Format content
            if (typeof rawData === 'object' && rawData !== null) {
                contentEl.textContent = JSON.stringify(rawData, null, 2);
            } else {
                contentEl.textContent = rawData;
            }
            
            // Show container
            detailsContainer.classList.remove('hidden');
        }

        function showTab(ticker) {
            // Hide all content blocks
            document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
            
            // Show the selected content block
            const content = document.getElementById('content-' + ticker);
            if(content) content.classList.remove('hidden');

            // Reset tab button styling
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('bg-indigo-600', 'text-white', 'shadow-md');
                btn.classList.add('text-gray-600', 'hover:bg-gray-100');
            });
            
            // Activate selected tab button styling
            const activeBtn = document.getElementById('tab-' + ticker);
            if(activeBtn) {
                activeBtn.classList.remove('text-gray-600', 'hover:bg-gray-100');
                activeBtn.classList.add('bg-indigo-600', 'text-white', 'shadow-md');
            }
        }
        
        function filterTabs() {
            const search = document.getElementById('search').value.toLowerCase();
            document.querySelectorAll('.tab-item').forEach(li => {
                const t = li.dataset.ticker.toLowerCase();
                if (t.includes(search)) {
                    li.style.display = 'block';
                } else {
                    li.style.display = 'none';
                }
            });
        }

        document.addEventListener("DOMContentLoaded", () => {
            // Render markdown content
            document.querySelectorAll('.markdown-content').forEach(el => {
                const rawMarkdown = el.textContent;
                const container = document.createElement('div');
                container.innerHTML = marked.parse(rawMarkdown);
                container.className = 'prose prose-indigo max-w-none';
                el.parentNode.insertBefore(container, el);
                el.style.display = 'none';
            });

            // Auto-click the Compare tab on load
            const compareBtn = document.getElementById('tab-Compare');
            if(compareBtn) compareBtn.click();
            else {
                const firstBtn = document.querySelector('.tab-btn');
                if(firstBtn) firstBtn.click();
            }
        });
    </script>
    <style>
        .fade-in { animation: fadeIn 0.3s ease-in-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        /* Custom scrollbar for pre tags and custom-scrollbar class */
        pre::-webkit-scrollbar, .custom-scrollbar::-webkit-scrollbar { width: 8px; height: 8px; }
        pre::-webkit-scrollbar-track, .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        pre::-webkit-scrollbar-thumb, .custom-scrollbar::-webkit-scrollbar-thumb { background: #4b5563; border-radius: 4px; }
    </style>
</head>
<body class="bg-gray-100 text-gray-900 font-sans h-screen flex overflow-hidden">
    
    <!-- Sidebar / Tab Bar -->
    <div class="w-72 bg-white border-r border-gray-200 flex flex-col h-full shadow-sm z-10 shrink-0">
        <div class="p-5 border-b border-gray-200 bg-gray-50">
            <h1 class="text-xl font-black text-gray-900 tracking-tight flex items-center">
                <svg class="w-6 h-6 mr-2 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>
                Buffett AI
            </h1>
            <input type="text" id="search" onkeyup="filterTabs()" placeholder="Search tickers..." class="mt-4 w-full px-4 py-2 bg-white border border-gray-300 rounded-lg text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all">
        </div>
        <ul class="flex-1 overflow-y-auto p-3 space-y-1">
            <li class="tab-item" data-ticker="About">
                <button onclick="showTab('About')" id="tab-About" class="tab-btn w-full text-left px-4 py-3 rounded-xl font-bold transition-all duration-150 text-gray-600 hover:bg-gray-100">
                    About
                </button>
            </li>
            <li class="tab-item" data-ticker="Principles">
                <button onclick="showTab('Principles')" id="tab-Principles" class="tab-btn w-full text-left px-4 py-3 rounded-xl font-bold transition-all duration-150 text-gray-600 hover:bg-gray-100">
                    Principles
                </button>
            </li>
            <li class="tab-item" data-ticker="Compare">
                <button onclick="showTab('Compare')" id="tab-Compare" class="tab-btn w-full text-left px-4 py-3 rounded-xl font-bold transition-all duration-150 text-gray-600 hover:bg-gray-100">
                    Compare
                </button>
            </li>
            {% for ticker in reports.keys()|sort %}
            <li class="tab-item" data-ticker="{{ ticker }}">
                <button onclick="showTab('{{ ticker }}')" id="tab-{{ ticker }}" class="tab-btn w-full text-left px-4 py-3 rounded-xl font-bold transition-all duration-150 text-gray-600 hover:bg-gray-100">
                    {{ ticker }}
                </button>
            </li>
            {% endfor %}
        </ul>
    </div>

    <!-- Main Content Area -->
    <div class="flex-1 h-full overflow-y-auto bg-gray-50/50 p-8 sm:p-12 relative">
        <div id="content-About" class="tab-content hidden max-w-6xl mx-auto pb-20 fade-in">
            <div class="border-b border-gray-200 pb-6 mb-8">
                <h2 class="text-5xl font-black text-gray-900 tracking-tight">About Buffett AI</h2>
                <h3 class="text-xl font-bold text-gray-600 mt-2">The 8 Principles of Warren Buffett</h3>
            </div>
            
            <p class="text-gray-700 text-xl leading-relaxed mb-10 max-w-4xl">
                This tool evaluates companies using the heuristics in the Buffett evaluator pipeline.
                Below is the raw reference documentation for the company-analysis side of the system.
            </p>

            <div class="space-y-6 max-w-5xl">
                {% for md_file in markdown_files %}
                <details class="group bg-white shadow-sm border border-gray-200 rounded-2xl overflow-hidden transition-all duration-300 open:shadow-lg open:ring-1 open:ring-indigo-500/20">
                    <summary class="cursor-pointer px-6 py-5 bg-white hover:bg-gray-50 transition-colors duration-200 flex items-center justify-between select-none">
                        <div>
                            <h4 class="text-xl font-bold text-gray-900">{{ md_file.name | replace('.md', '') | replace('-', ' ') | title }}</h4>
                        </div>
                        <span class="ml-4 flex-shrink-0 bg-gray-100 rounded-full p-2 group-open:bg-indigo-100 group-open:text-indigo-600 text-gray-400 transition-colors">
                            <svg class="w-5 h-5 group-open:rotate-180 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                        </span>
                    </summary>
                    <div class="border-t border-gray-100 bg-white">
                        <div class="p-8 overflow-x-auto">
                            <div class="markdown-content hidden">{{ md_file.content }}</div>
                        </div>
                    </div>
                </details>
                {% endfor %}
            </div>
        </div>

        <div id="content-Principles" class="tab-content hidden max-w-6xl mx-auto pb-20 fade-in">
            <div class="border-b border-gray-200 pb-6 mb-8">
                <h2 class="text-5xl font-black text-gray-900 tracking-tight">Investor Principles</h2>
                <h3 class="text-xl font-bold text-gray-600 mt-2">Notes to self for position sizing, holding, and behavior</h3>
            </div>

            <p class="text-gray-700 text-xl leading-relaxed mb-10 max-w-4xl">
                These are investor principles rather than company evaluators. They guide how to choose, hold, size, and avoid behavioral mistakes.
            </p>

            <div class="space-y-6 max-w-5xl">
                {% for md_file in principle_files %}
                <details class="group bg-white shadow-sm border border-gray-200 rounded-2xl overflow-hidden transition-all duration-300 open:shadow-lg open:ring-1 open:ring-indigo-500/20">
                    <summary class="cursor-pointer px-6 py-5 bg-white hover:bg-gray-50 transition-colors duration-200 flex items-center justify-between select-none">
                        <div>
                            <h4 class="text-xl font-bold text-gray-900">{{ md_file.name | replace('.md', '') | replace('-', ' ') | title }}</h4>
                        </div>
                        <span class="ml-4 flex-shrink-0 bg-gray-100 rounded-full p-2 group-open:bg-indigo-100 group-open:text-indigo-600 text-gray-400 transition-colors">
                            <svg class="w-5 h-5 group-open:rotate-180 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                        </span>
                    </summary>
                    <div class="border-t border-gray-100 bg-white">
                        <div class="p-8 overflow-x-auto">
                            <div class="markdown-content hidden">{{ md_file.content }}</div>
                        </div>
                    </div>
                </details>
                {% endfor %}
            </div>
        </div>

        <div id="content-Compare" class="tab-content hidden max-w-6xl mx-auto pb-20 fade-in">
            <div class="border-b border-gray-200 pb-6 mb-8">
                <h2 class="text-5xl font-black text-gray-900 tracking-tight">Compare</h2>
                <h3 class="text-xl font-bold text-gray-600 mt-2">Normalized rankings, qualitative risks, and heuristic scores across analyzed assets</h3>
            </div>

            <div class="space-y-8">
                {% if comparison_reports %}
                <div class="space-y-8">
                    {% for comparison_name, comparison in comparison_reports.items() %}
                    <section class="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 sm:p-8">
                        <div class="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between border-b border-gray-100 pb-5 mb-6">
                            <div>
                                <h3 class="text-3xl font-black text-gray-900 tracking-tight">{{ comparison_name | replace('_', ' ') | replace(' vs ', ' vs ') | title }}</h3>
                                <p class="text-sm font-semibold text-gray-500 mt-2">
                                    {{ comparison.get('company_count', 0) }} companies
                                    {% if comparison.get('generated_at') %}
                                    <span class="mx-2 text-gray-300">/</span>
                                    {{ comparison.get('generated_at') }}
                                    {% endif %}
                                </p>
                            </div>
                        </div>

                        <div class="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-8">
                            <div class="bg-gray-50 rounded-xl border border-gray-200 p-5">
                                <h4 class="text-lg font-bold text-gray-900 mb-4">Overall Rankings</h4>
                                <div class="space-y-3">
                                    {% for row in comparison.get('rankings', []) %}
                                    <div class="flex items-center justify-between bg-white rounded-lg border border-gray-200 px-4 py-3">
                                        <div>
                                            <div class="text-sm font-black text-gray-900">#{{ row.get('rank') }} {{ row.get('ticker') }}</div>
                                            <div class="text-xs text-gray-500">{{ row.get('company_name') }}</div>
                                        </div>
                                        <div class="text-right">
                                            <div class="text-sm font-bold text-indigo-700">{{ row.get('overall_score', 'N/A') }}</div>
                                            <div class="text-xs text-gray-500">Confidence {{ row.get('confidence_score', 'N/A') }}</div>
                                        </div>
                                    </div>
                                    {% endfor %}
                                </div>
                            </div>

                            <div class="bg-gray-50 rounded-xl border border-gray-200 p-5">
                                <h4 class="text-lg font-bold text-gray-900 mb-4">Category Leaders</h4>
                                <div class="space-y-3">
                                    {% for category, rows in comparison.get('category_rankings', {}).items() %}
                                    {% set leader = rows[0] if rows else None %}
                                    {% if leader %}
                                    <div class="flex items-center justify-between bg-white rounded-lg border border-gray-200 px-4 py-3">
                                        <div>
                                            <div class="text-sm font-bold text-gray-900">{{ category | camel_to_spaces | title }}</div>
                                            <div class="text-xs text-gray-500">Leader: {{ leader.get('ticker') }}</div>
                                        </div>
                                        <div class="text-sm font-bold text-emerald-700">{{ leader.get('score') }}</div>
                                    </div>
                                    {% endif %}
                                    {% endfor %}
                                </div>
                            </div>
                        </div>

                        <div class="mb-8">
                            <h4 class="text-lg font-bold text-gray-900 mb-4">Qualitative Risk Themes</h4>
                            <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
                                {% for company in comparison.get('companies', []) %}
                                {% set inversion = company.get('qualitative_scores', {}).get('inversion', {}) %}
                                {% set management = company.get('qualitative_scores', {}).get('management_sentiment', {}) %}
                                <div class="bg-gray-50 rounded-xl border border-gray-200 p-5">
                                    <div class="flex items-start justify-between mb-4">
                                        <div>
                                            <h5 class="text-lg font-black text-gray-900">{{ company.get('ticker') }}</h5>
                                            <p class="text-sm text-gray-500">{{ company.get('company_name') }}</p>
                                        </div>
                                        <div class="text-right text-sm">
                                            <div class="font-bold text-indigo-700">Overall {{ company.get('overall_score', 'N/A') }}</div>
                                            <div class="text-gray-500">Confidence {{ company.get('confidence_score', 'N/A') }}</div>
                                        </div>
                                    </div>

                                    <div class="grid grid-cols-2 gap-3 mb-4">
                                        <div class="bg-white rounded-lg border border-gray-200 p-3">
                                            <div class="text-xs uppercase tracking-wide text-gray-500">Inversion Risk</div>
                                            <div class="text-xl font-black text-rose-700 mt-1">{{ inversion.get('composite_risk_score', 'N/A') }}</div>
                                        </div>
                                        <div class="bg-white rounded-lg border border-gray-200 p-3">
                                            <div class="text-xs uppercase tracking-wide text-gray-500">Mgmt Sentiment</div>
                                            <div class="text-xl font-black text-emerald-700 mt-1">{{ management.get('sentiment_score', 'N/A') }}</div>
                                        </div>
                                    </div>

                                    {% if inversion.get('theme_scores') %}
                                    <div class="space-y-2">
                                        {% for theme, score in inversion.get('theme_scores', {}).items() %}
                                        <div>
                                            <div class="flex justify-between text-xs font-semibold text-gray-600 mb-1">
                                                <span>{{ theme | replace('_', ' ') | title }}</span>
                                                <span>{{ score }}</span>
                                            </div>
                                            <div class="w-full bg-gray-200 rounded-full h-2.5 overflow-hidden">
                                                <div class="bg-rose-500 h-2.5 rounded-full" style="width: {{ score }}%"></div>
                                            </div>
                                        </div>
                                        {% endfor %}
                                    </div>
                                    {% endif %}
                                </div>
                                {% endfor %}
                            </div>
                        </div>

                        <div class="mb-8">
                            <h4 class="text-lg font-bold text-gray-900 mb-4">Category Scores By Company</h4>
                            <div class="overflow-x-auto">
                                <table class="min-w-full text-sm bg-white border border-gray-200 rounded-xl overflow-hidden">
                                    <thead class="bg-gray-50 text-gray-600 uppercase tracking-wide text-xs">
                                        <tr>
                                            <th class="px-4 py-3 text-left">Ticker</th>
                                            {% for category in comparison.get('category_rankings', {}).keys() %}
                                            <th class="px-4 py-3 text-left">{{ category | camel_to_spaces | title }}</th>
                                            {% endfor %}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {% for company in comparison.get('companies', []) %}
                                        <tr class="border-t border-gray-100">
                                            <td class="px-4 py-3 font-bold text-gray-900">{{ company.get('ticker') }}</td>
                                            {% for category in comparison.get('category_rankings', {}).keys() %}
                                            <td class="px-4 py-3 text-gray-700">{{ company.get('category_scores', {}).get(category, 'N/A') }}</td>
                                            {% endfor %}
                                        </tr>
                                        {% endfor %}
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        <details class="text-sm text-gray-500">
                            <summary class="cursor-pointer hover:text-indigo-600 font-semibold">Raw Comparison Data</summary>
                            <pre class="mt-3 text-xs bg-gray-900 text-emerald-400 p-4 rounded overflow-x-auto">{{ comparison | tojson(indent=2) }}</pre>
                        </details>
                    </section>
                    {% endfor %}
                </div>
                {% endif %}

                {% if reports %}
                <div>
                    <div class="flex items-end justify-between mb-4">
                        <h3 class="text-2xl font-black text-gray-900">Heuristic Heatmap</h3>
                        <p class="text-sm text-gray-500">Quick polarity scan of scorable heuristic outputs</p>
                    </div>
                {% for ticker, data in reports.items() %}
                <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                    <div class="flex justify-between items-end mb-4 border-b border-gray-100 pb-2">
                        <div>
                            <h3 class="text-2xl font-black text-gray-900 tracking-tight">{{ ticker }}</h3>
                            <p class="text-sm font-bold text-gray-500">{{ data.get('company_name', ticker) }}</p>
                        </div>
                        <button onclick="showTab('{{ ticker }}')" class="text-sm font-bold text-indigo-600 hover:text-indigo-800 transition-colors">
                            View Details &rarr;
                        </button>
                    </div>
                    
                    <div class="flex flex-wrap gap-1">
                        {% for score_obj in data.get('heatmap_scores', []) %}
                            {% set s = score_obj.score %}
                            {% if s == -999.0 %} 
                                {% set color_class = "bg-white border-2 border-dashed border-gray-300 opacity-60 hover:opacity-100" %}
                                {% set label_text = score_obj.name ~ " (N/A)" %}
                            {% elif s < -0.5 %} 
                                {% set color_class = "bg-rose-600" %}
                                {% set label_text = score_obj.name ~ " (Score: " ~ '%.2f'|format(s) ~ ")" %}
                            {% elif s < -0.1 %} 
                                {% set color_class = "bg-rose-400" %}
                                {% set label_text = score_obj.name ~ " (Score: " ~ '%.2f'|format(s) ~ ")" %}
                            {% elif s <= 0.0 %} 
                                {% set color_class = "bg-gray-200" %}
                                {% set label_text = score_obj.name ~ " (Score: " ~ '%.2f'|format(s) ~ ")" %}
                            {% elif s < 0.5 %} 
                                {% set color_class = "bg-emerald-400" %}
                                {% set label_text = score_obj.name ~ " (Score: " ~ '%.2f'|format(s) ~ ")" %}
                            {% else %} 
                                {% set color_class = "bg-emerald-600" %}
                                {% set label_text = score_obj.name ~ " (Score: " ~ '%.2f'|format(s) ~ ")" %}
                            {% endif %}
                            
                            <button onclick="showHeuristicDetails('{{ ticker }}', '{{ score_obj.category }}', '{{ score_obj.heuristic }}')" class="w-6 h-6 sm:w-8 sm:h-8 rounded-sm {{ color_class }} hover:ring-2 hover:ring-offset-1 hover:ring-indigo-500 cursor-pointer transition-all relative group focus:outline-none focus:ring-2 focus:ring-indigo-600">
                                <div class="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 hidden group-hover:block bg-gray-900 text-white text-xs py-1 px-2 rounded whitespace-nowrap z-10 shadow-lg pointer-events-none">
                                    {{ label_text }}
                                </div>
                            </button>
                        {% endfor %}
                    </div>
                    
                    <div id="details-{{ ticker }}" class="hidden mt-6 bg-gray-900 rounded-xl overflow-hidden shadow-inner border border-gray-800 transition-all">
                        <div class="px-4 py-3 bg-gray-800 border-b border-gray-700 flex justify-between items-center">
                            <h4 id="details-title-{{ ticker }}" class="font-bold text-gray-200 text-sm tracking-wide uppercase"></h4>
                            <button onclick="document.getElementById('details-{{ ticker }}').classList.add('hidden')" class="text-gray-400 hover:text-white transition-colors focus:outline-none">
                                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                            </button>
                        </div>
                        <div class="p-4 overflow-x-auto max-h-96 overflow-y-auto custom-scrollbar">
                            <pre id="details-content-{{ ticker }}" class="text-emerald-400 font-mono text-sm whitespace-pre-wrap bg-transparent p-0 m-0 leading-relaxed"></pre>
                        </div>
                    </div>
                </div>
                {% endfor %}
                <p class="text-xs text-gray-400 mt-2 text-center">Each box represents one specific framework/heuristic. Color corresponds to normalized polarity: Red (Negative/Risk), Gray (Neutral/Raw), Green (Positive/Advantage).</p>
                </div>
                {% endif %}

                {% if not comparison_reports and not reports %}
                <div class="text-center py-16 px-10 bg-white shadow-xl rounded-3xl max-w-md border border-gray-100 mx-auto">
                    <p class="text-gray-500 text-xl font-medium mb-3">No analysis or comparison files found.</p>
                    <p class="text-gray-400">Run the pipeline first to generate data in <code class="bg-gray-100 px-2 py-1 rounded text-sm text-gray-600">output/</code></p>
                </div>
                {% endif %}
            </div>
        </div>

        {% if not reports and not comparison_reports %}
        <div class="flex items-center justify-center h-full">
            <div class="text-center py-16 px-10 bg-white shadow-xl rounded-3xl max-w-md border border-gray-100">
                <p class="text-gray-500 text-xl font-medium mb-3">No analysis files found.</p>
                <p class="text-gray-400">Run the pipeline first to generate data in <code class="bg-gray-100 px-2 py-1 rounded text-sm text-gray-600">output/</code></p>
            </div>
        </div>
        {% endif %}

        {% for ticker, data in reports.items() %}
        <div id="content-{{ ticker }}" class="tab-content hidden max-w-6xl mx-auto pb-20 fade-in">
            
            <div class="border-b border-gray-200 pb-6 mb-8 flex justify-between items-end">
                <div>
                    <h2 class="text-5xl font-black text-gray-900 tracking-tight">{{ ticker }}</h2>
                    <h3 class="text-xl font-bold text-gray-600 mt-2">{{ data.get('company_name', ticker) }}</h3>
                </div>
            </div>

            <div class="mb-10">
                <h3 class="text-lg font-bold text-gray-800 mb-3 uppercase tracking-wide">Business Overview</h3>
                <p class="text-gray-600 leading-relaxed">{{ data.get('description', 'N/A') }}</p>
            </div>
            
            <h3 class="text-2xl font-black text-gray-900 mb-6 border-b border-gray-200 pb-3">49 Heuristics Evaluations</h3>

            <div class="space-y-6">
                <!-- Group the outputs by the top-level keys in the JSON or iterate generally -->
                {% for category, results in data.items() %}
                    {% if category not in ['ticker', 'company_name', 'description'] and results is mapping %}
                        <div class="bg-gray-100 rounded-xl p-4 shadow-inner mb-6">
                            <h3 class="text-xl font-bold text-indigo-800 mb-4">{{ category | camel_to_spaces | title }}</h3>
                            
                            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {% for heuristic, details in results.items() %}
                                {% if details is mapping %}
                                <div class="bg-white border {% if details.get('pass') == true %} border-emerald-300 ring-1 ring-emerald-100 {% elif details.get('pass') == false %} border-rose-300 ring-1 ring-rose-100 {% else %} border-gray-200 {% endif %} rounded-lg p-5 shadow-sm">
                                    <div class="flex justify-between items-start mb-2">
                                        <h4 class="text-md font-bold text-gray-900">{{ heuristic | camel_to_spaces }}</h4>
                                        {% if details.get('score') %}
                                            <span class="px-2 py-1 text-xs font-bold rounded-full {% if details.get('pass') %} bg-emerald-100 text-emerald-700 {% else %} bg-rose-100 text-rose-700 {% endif %}">
                                                {{ details.get('score') }}/10
                                            </span>
                                        {% endif %}
                                    </div>
                                    
                                    {% if details.get('analysis') %}
                                        <p class="text-gray-700 text-sm leading-relaxed mb-3">{{ details.get('analysis') }}</p>
                                    {% endif %}
                                    
                                    {% if details.get('justification') %}
                                        <p class="text-gray-700 text-sm leading-relaxed mb-3">{{ details.get('justification') }}</p>
                                    {% endif %}
                                    
                                    <!-- Fallback to raw JSON dump if there's arbitrary structure (e.g. DCF Output, Leverage Risk Output) -->
                                    <details class="text-sm text-gray-500">
                                        <summary class="cursor-pointer hover:text-indigo-600">Raw Data</summary>
                                        <pre class="mt-2 text-xs bg-gray-900 text-emerald-400 p-2 rounded overflow-x-auto">{{ details | tojson(indent=2) }}</pre>
                                    </details>
                                </div>
                                {% else %}
                                    <!-- Some heuristics might just return a direct string or list -->
                                    <div class="bg-white border border-gray-200 rounded-lg p-5 shadow-sm">
                                        <h4 class="text-md font-bold text-gray-900 mb-2">{{ heuristic | camel_to_spaces }}</h4>
                                        <pre class="text-xs bg-gray-900 text-emerald-400 p-2 rounded overflow-x-auto">{{ details | tojson(indent=2) if details is not string else details }}</pre>
                                    </div>
                                {% endif %}
                            {% endfor %}
                            </div>
                        </div>
                    {% elif category not in ['ticker', 'company_name', 'description'] %}
                         <div class="bg-white border border-gray-200 rounded-lg p-5 shadow-sm mb-4">
                            <h4 class="text-md font-bold text-gray-900 mb-2">{{ category | camel_to_spaces | title }}</h4>
                            <pre class="text-xs bg-gray-900 text-emerald-400 p-2 rounded overflow-x-auto">{{ results | tojson(indent=2) if results is not string else results }}</pre>
                        </div>
                    {% endif %}
                {% endfor %}
            </div>

        </div>
        {% endfor %}
        
    </div>
</body>
</html>
"""

def calculate_heatmap_scores(data):
    scores = []

    def normalize_to_heatmap(score_100):
        return max(-1.0, min(1.0, (float(score_100) - 50.0) / 50.0))

    def extract_text_and_score(obj):
        text_content = ""
        direct_score = None
        
        if isinstance(obj, dict):
            # Explicit N/A Check
            if 'applicable' in obj and obj['applicable'] is False:
                return "", -999.0
                
            # Check for high-signal booleans/numbers
            if 'is_undervalued' in obj:
                direct_score = 1.0 if obj['is_undervalued'] else -1.0
            elif 'inside_circle' in obj:
                direct_score = 1.0 if obj['inside_circle'] else -1.0
            elif 'avoid_industry' in obj:
                direct_score = -1.0 if obj['avoid_industry'] else 1.0
            elif 'is_value_trap' in obj:
                direct_score = -1.0 if obj['is_value_trap'] else 1.0
            elif 'margin_of_safety' in obj and isinstance(obj['margin_of_safety'], (int, float)):
                # If margin > 0 it's good, scale up to 1.0 based on size
                val = float(obj['margin_of_safety'])
                direct_score = max(-1.0, min(1.0, val * 2)) 
            elif 'intrinsic_value_per_share' in obj and 'market_price' in obj:
                try:
                    iv = float(obj['intrinsic_value_per_share'])
                    mp = float(obj['market_price'])
                    direct_score = 1.0 if iv > mp else -1.0
                except:
                    pass
            elif 'goodwill_quality' in obj:
                val = str(obj['goodwill_quality']).lower()
                direct_score = 1.0 if 'economic' in val else -1.0
            elif any(k in obj for k in ['durability_assessment', 'technology_quality', 'consumer_brand_quality']):
                key = next(k for k in ['durability_assessment', 'technology_quality', 'consumer_brand_quality'] if k in obj)
                val = str(obj[key]).lower()
                if val == 'strong': direct_score = 1.0
                elif val == 'moderate' or val == 'mixed': direct_score = 0.0
                elif val == 'weak': direct_score = -1.0
            elif 'mr_market_mood' in obj:
                if 'contrarian_score' in obj and isinstance(obj['contrarian_score'], (int, float)):
                    direct_score = max(-1.0, min(1.0, float(obj['contrarian_score'])))
                else:
                    mood = str(obj['mr_market_mood']).lower()
                    if mood == 'fear': direct_score = 0.8
                    elif mood == 'greed': direct_score = -0.8
                    elif mood == 'neutral': direct_score = 0.0
            elif 'profit_margin' in obj and isinstance(obj['profit_margin'], (int, float)):
                val = float(obj['profit_margin'])
                direct_score = max(-1.0, min(1.0, val * 5)) # 20% margin = 1.0
            elif 'owner_earnings' in obj and isinstance(obj['owner_earnings'], (int, float)):
                direct_score = 1.0 if float(obj['owner_earnings']) > 0 else -1.0
            elif 'capital_allocation_discipline' in obj:
                disc = str(obj['capital_allocation_discipline']).lower()
                if disc == 'strong': direct_score = 1.0
                elif disc == 'moderate': direct_score = 0.5
                elif disc == 'weak': direct_score = -1.0
                elif disc == 'neutral': direct_score = 0.0
            elif 'acquisition_discipline' in obj:
                disc = str(obj['acquisition_discipline']).lower()
                if disc == 'strong': direct_score = 1.0
                elif disc == 'mixed': direct_score = 0.0
                elif disc == 'weak': direct_score = -1.0
            elif 'return_on_invested_capital' in obj and isinstance(obj['return_on_invested_capital'], (int, float)):
                val = float(obj['return_on_invested_capital'])
                direct_score = max(-1.0, min(1.0, val * 4)) # 25% ROIC = 1.0
            elif 'verdict' in obj:
                verdict = str(obj['verdict']).lower()
                if 'risky' in verdict or 'fail' in verdict or 'avoid' in verdict:
                    direct_score = -1.0
                elif 'pass' in verdict or 'favorable' in verdict or 'investable' in verdict:
                    direct_score = 1.0
            elif 'debt_funded' in obj:
                direct_score = -1.0 if obj['debt_funded'] else 1.0
            elif 'economic_reality_assessment' in obj:
                val = str(obj['economic_reality_assessment']).lower()
                direct_score = 1.0 if 'cash' in val else -1.0
            elif 'shareholder_orientation' in obj:
                val = str(obj['shareholder_orientation']).lower()
                if val == 'strong': direct_score = 1.0
                elif val == 'moderate': direct_score = 0.5
                elif val == 'weak': direct_score = -1.0
            elif 'buyback_strategy' in obj:
                val = str(obj['buyback_strategy']).lower()
                if 'value' in val or 'opportunistic' in val: direct_score = 1.0
                else: direct_score = -0.5
            elif 'retained_value_creation' in obj and isinstance(obj['retained_value_creation'], (int, float)):
                val = float(obj['retained_value_creation'])
                direct_score = max(-1.0, min(1.0, val * 5))
            elif 'derivatives_risk' in obj:
                val = str(obj['derivatives_risk']).lower()
                if val == 'low': direct_score = 1.0
                elif val == 'high': direct_score = -1.0
                else: direct_score = 0.0
            elif 'leverage_assessment' in obj:
                val = str(obj['leverage_assessment']).lower()
                if val == 'low': direct_score = 1.0
                elif val == 'high': direct_score = -1.0
                else: direct_score = 0.0
            elif skill_name == 'LeverageRisk':
                if 'toxic_derivative_exposure' in obj:
                    val = str(obj['toxic_derivative_exposure']).lower()
                    if 'none' in val or 'no ' in val: direct_score = 0.5
                    elif 'high' in val or 'toxic' in val: direct_score = -1.0
                    else: direct_score = 0.0
            elif 'business_model_type' in obj:
                val = str(obj['business_model_type']).lower()
                direct_score = 1.0 if val in ['recurring', 'hybrid', 'subscription', 'software', 'franchise'] else -0.5
            elif 'moat_type' in obj:
                val = str(obj['moat_type']).lower()
                direct_score = 1.0 if val != 'none' and val != 'no moat' else -1.0

            if direct_score is None:
                normalized_candidates = []
                for k, v in obj.items():
                    if isinstance(v, bool):
                        score_100 = comparison_scoring._score_boolean(k, v)
                        if score_100 is not None:
                            normalized_candidates.append(normalize_to_heatmap(score_100))
                    elif isinstance(v, (int, float)):
                        score_100 = comparison_scoring._score_fixed_scale(k, v)
                        if score_100 is not None:
                            normalized_candidates.append(normalize_to_heatmap(score_100))
                    elif isinstance(v, str):
                        score_100 = comparison_scoring._score_label(v)
                        if score_100 is not None:
                            normalized_candidates.append(normalize_to_heatmap(score_100))
                if normalized_candidates:
                    direct_score = sum(normalized_candidates) / len(normalized_candidates)

            for k, v in obj.items():
                if isinstance(v, str):
                    text_content += v + " "
                elif isinstance(v, (dict, list)):
                    sub_text, _ = extract_text_and_score(v)
                    text_content += sub_text + " "
                    
        elif isinstance(obj, list):
            # Special check for Inflation arrays
            if len(obj) > 0 and isinstance(obj[0], dict) and 'Pricing_Power_Assessment' in obj[0]:
                has_strong = any(str(item.get('Pricing_Power_Assessment', '')).lower().startswith('strong') for item in obj)
                has_weak = any(str(item.get('Pricing_Power_Assessment', '')).lower().startswith('weak') for item in obj)
                if has_strong:
                    direct_score = 1.0
                elif has_weak:
                    direct_score = -1.0
                else:
                    direct_score = 0.0

            for item in obj:
                sub_text, _ = extract_text_and_score(item)
                text_content += sub_text + " "
        elif isinstance(obj, str):
            text_content += obj + " "
            
        return text_content.lower(), direct_score

    for category, results in data.items():
        if category in ['ticker', 'company_name', 'description'] or not isinstance(results, dict):
            continue
            
        for skill_name, skill_content in results.items():
            text_content, direct_score = extract_text_and_score(skill_content)
            
            if direct_score is not None:
                score = direct_score
            else:
                if text_content.strip() == "raw data" or text_content.strip() == "":
                    score = None
                else:
                    score = None
                
            scores.append({
                "name": camel_to_spaces(skill_name).title(), 
                "score": score if score is not None else -999.0,
                "category": category,
                "heuristic": skill_name
            })
            
    return scores

@app.route('/')
def index():
    reports = {}
    for file_path in glob.glob(str(OUTPUT_DIR / "*_analysis.json")):
        ticker = os.path.basename(file_path).split("_")[0]
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                data['heatmap_scores'] = calculate_heatmap_scores(data)
                reports[ticker] = data
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse {file_path}. Skipping. Error: {e}")
            continue

    comparison_reports = {}
    for file_path in glob.glob(str(OUTPUT_DIR / "*_comparison.json")):
        comparison_name = os.path.basename(file_path).replace("_comparison.json", "")
        try:
            with open(file_path, "r") as f:
                comparison_reports[comparison_name] = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse {file_path}. Skipping. Error: {e}")
            continue

    if reports and not comparison_reports:
        analyses = list(reports.values())
        comparison_reports["live_compare"] = comparison_scoring.build_comparison(analyses)

    # Load Markdown files for the About section
    markdown_files = []
    md_paths = sorted(glob.glob(str(REFERENCES_DIR / "*.md")))
    for md_path in md_paths:
        with open(md_path, "r", encoding="utf-8") as f:
            markdown_files.append({
                "name": os.path.basename(md_path),
                "content": f.read()
            })

    principle_files = []
    principle_paths = sorted(glob.glob(str(PRINCIPLES_DIR / "*.md")))
    for md_path in principle_paths:
        with open(md_path, "r", encoding="utf-8") as f:
            principle_files.append({
                "name": os.path.basename(md_path),
                "content": f.read()
            })
            
    return render_template_string(
        TEMPLATE,
        reports=reports,
        comparison_reports=dict(sorted(comparison_reports.items())),
        markdown_files=markdown_files,
        principle_files=principle_files,
    )

if __name__ == "__main__":
    print("Starting Buffett Skills Viewer on http://127.0.0.1:5052")
    app.run(port=5052, debug=True)
