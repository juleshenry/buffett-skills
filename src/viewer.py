from pathlib import Path
from flask import Flask, render_template_string
import glob, json, os

app = Flask(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"
REFERENCES_DIR = ROOT_DIR / "skills" / "buffett" / "references"

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

            // Auto-click the first tab on load
            const firstBtn = document.querySelector('.tab-btn');
            if(firstBtn) firstBtn.click();
        });
    </script>
    <style>
        .fade-in { animation: fadeIn 0.3s ease-in-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        /* Custom scrollbar for pre tags */
        pre::-webkit-scrollbar { width: 8px; height: 8px; }
        pre::-webkit-scrollbar-track { background: transparent; }
        pre::-webkit-scrollbar-thumb { background: #4b5563; border-radius: 4px; }
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
                This tool evaluates companies based on 49 core principles derived from Warren Buffett's investment philosophy. 
                Below is the comprehensive raw documentation detailing each of these principles and the exact analytical 
                playbooks they are based on.
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

        {% if not reports %}
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

@app.route('/')
def index():
    reports = {}
    for file_path in glob.glob(str(OUTPUT_DIR / "*_analysis.json")):
        ticker = os.path.basename(file_path).split("_")[0]
        try:
            with open(file_path, "r") as f:
                reports[ticker] = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse {file_path}. Skipping. Error: {e}")
            continue
    
    # Load Markdown files for the About section
    markdown_files = []
    md_paths = sorted(glob.glob(str(REFERENCES_DIR / "*.md")))
    for md_path in md_paths:
        with open(md_path, "r", encoding="utf-8") as f:
            markdown_files.append({
                "name": os.path.basename(md_path),
                "content": f.read()
            })
            
    return render_template_string(TEMPLATE, reports=reports, markdown_files=markdown_files)

if __name__ == "__main__":
    print("Starting Buffett Skills Viewer on http://127.0.0.1:5052")
    app.run(port=5052, debug=True)
