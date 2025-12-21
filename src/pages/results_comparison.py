"""
Results & Comparison Component for AES S-box Construction
Based on the paper: "AES S-box modification uses affine matrices exploration"

This module provides comprehensive comparison and analysis of S-boxes:
1. Multiple S-box management
2. Side-by-side comparison
3. Statistical analysis
4. Performance ranking
5. Export capabilities
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class ResultsManager:
    """
    Class to manage and compare multiple S-box results.
    """
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state for storing S-boxes."""
        if 'saved_sboxes' not in st.session_state:
            st.session_state.saved_sboxes = {}
        if 'comparison_selection' not in st.session_state:
            st.session_state.comparison_selection = []
    
    def save_sbox(self, name: str, sbox: np.ndarray, 
                  test_results: Dict, metadata: Optional[Dict] = None):
        """
        Save S-box with test results.
        
        Args:
            name: Name for the S-box
            sbox: 16x16 S-box array
            test_results: Dictionary of test results
            metadata: Optional metadata (affine matrix, constant, etc.)
        """
        st.session_state.saved_sboxes[name] = {
            'sbox': sbox.copy(),
            'results': test_results.copy(),
            'metadata': metadata if metadata else {},
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def delete_sbox(self, name: str):
        """Delete saved S-box."""
        if name in st.session_state.saved_sboxes:
            del st.session_state.saved_sboxes[name]
    
    def get_sbox(self, name: str) -> Optional[Dict]:
        """Get saved S-box data."""
        return st.session_state.saved_sboxes.get(name)
    
    def list_sboxes(self) -> List[str]:
        """Get list of saved S-box names."""
        return list(st.session_state.saved_sboxes.keys())
    
    def compare_sboxes(self, names: List[str]) -> pd.DataFrame:
        """
        Compare multiple S-boxes.
        
        Args:
            names: List of S-box names to compare
            
        Returns:
            DataFrame with comparison data
        """
        metrics = ['NL', 'SAC', 'BIC-NL', 'BIC-SAC', 'LAP', 'DAP', 
                  'DU', 'AD', 'TO', 'CI']
        
        comparison_data = {'Metric': metrics}
        
        for name in names:
            sbox_data = self.get_sbox(name)
            if sbox_data:
                results = sbox_data['results']
                comparison_data[name] = [
                    results['NL']['value'],
                    results['SAC']['value'],
                    results['BIC-NL']['value'],
                    results['BIC-SAC']['value'],
                    results['LAP']['value'],
                    results['DAP']['value'],
                    results['DU']['value'],
                    results['AD']['value'],
                    results['TO']['value'],
                    results['CI']['value']
                ]
        
        return pd.DataFrame(comparison_data)
    
    def calculate_ranking(self) -> pd.DataFrame:
        """
        Rank all saved S-boxes by strength.
        
        Returns:
            DataFrame with rankings
        """
        if not st.session_state.saved_sboxes:
            return pd.DataFrame()
        
        ranking_data = []
        
        for name, data in st.session_state.saved_sboxes.items():
            results = data['results']
            
            # Calculate strength value
            sv = (120 - results['NL']['value']) + \
                 abs(0.5 - results['SAC']['value']) + \
                 (120 - results['BIC-NL']['value']) + \
                 abs(0.5 - results['BIC-SAC']['value'])
            
            # Extended score
            extended = sv + \
                      (results['DU']['value'] - 4) + \
                      (7 - results['AD']['value']) + \
                      results['TO']['value']
            
            # Count excellent criteria
            excellent = sum([
                results['NL']['value'] >= 112,
                abs(results['SAC']['value'] - 0.5) <= 0.01,
                results['BIC-NL']['value'] >= 112,
                abs(results['BIC-SAC']['value'] - 0.5) <= 0.01,
                results['LAP']['value'] <= 0.0625,
                results['DAP']['value'] <= 0.015625,
                results['DU']['value'] <= 4,
                results['AD']['value'] >= 7,
                results['TO']['value'] <= 0.1,
                results['CI']['value'] >= 0
            ])
            
            ranking_data.append({
                'S-box Name': name,
                'SV': sv,
                'Extended Score': extended,
                'Excellent Criteria': f"{excellent}/10",
                'NL': results['NL']['value'],
                'SAC': results['SAC']['value'],
                'DU': results['DU']['value'],
                'AD': results['AD']['value'],
                'Timestamp': data['timestamp']
            })
        
        df = pd.DataFrame(ranking_data)
        df = df.sort_values('SV')  # Lower SV is better
        df.insert(0, 'Rank', range(1, len(df) + 1))
        
        return df


def render_results_comparison():
    """
    Streamlit component for results comparison and analysis.
    """
    st.header("📊 Results & Comparison")
    
    # Initialize manager
    manager = ResultsManager()
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💾 Save Current S-box",
        "📋 Saved S-boxes",
        "⚖️ Side-by-Side Comparison",
        "🏆 Rankings",
        "📈 Statistical Analysis"
    ])
    
    # Tab 1: Save Current S-box
    with tab1:
        st.subheader("Save Current S-box")
        
        # Check if there's a current S-box and test results
        has_sbox = hasattr(st.session_state, 'constructed_sbox')
        has_results = hasattr(st.session_state, 'test_results')
        
        if not has_sbox:
            st.warning("⚠️ No S-box constructed. Please construct an S-box first.")
        elif not has_results:
            st.warning("⚠️ S-box not tested yet. Please run tests first.")
        else:
            st.success("✅ Current S-box and test results available")
            
            # Display current S-box
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Current S-box (Full Table):**")
                sbox_df = pd.DataFrame(
                    st.session_state.constructed_sbox,
                    columns=[f"{i:X}" for i in range(16)],
                    index=[f"{i:X}" for i in range(16)]
                )
                st.dataframe(sbox_df, width="stretch")
            
            with col2:
                st.write("**Test Results Summary:**")
                results = st.session_state.test_results
                
                # Create metrics dataframe
                metrics_data = {
                    'Metric': ['NL', 'SAC', 'BIC-NL', 'BIC-SAC', 'LAP', 'DAP', 'DU', 'AD', 'TO', 'CI'],
                    'Value': [
                        float(results["NL"]["value"]),
                        float(results["SAC"]["value"]),
                        float(results["BIC-NL"]["value"]),
                        float(results["BIC-SAC"]["value"]),
                        float(results["LAP"]["value"]),
                        float(results["DAP"]["value"]),
                        float(results["DU"]["value"]),
                        float(results["AD"]["value"]),
                        float(results["TO"]["value"]),
                        float(results["CI"]["value"]),
                    ]
                }
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, hide_index=True, width="stretch")
                
                # Calculate and show SV
                sv = (120 - results['NL']['value']) + \
                     abs(0.5 - results['SAC']['value']) + \
                     (120 - results['BIC-NL']['value']) + \
                     abs(0.5 - results['BIC-SAC']['value'])
                
                st.metric("Strength Value (SV)", f"{sv:.6f}")
                
                # Count excellent criteria
                excellent_count = sum([
                    results['NL']['value'] >= 112,
                    abs(results['SAC']['value'] - 0.5) <= 0.01,
                    results['BIC-NL']['value'] >= 112,
                    abs(results['BIC-SAC']['value'] - 0.5) <= 0.01,
                    results['LAP']['value'] <= 0.0625,
                    results['DAP']['value'] <= 0.015625,
                    results['DU']['value'] <= 4,
                    results['AD']['value'] >= 7,
                    results['TO']['value'] <= 0.1,
                    results['CI']['value'] >= 0
                ])
                
                st.metric("Excellent Criteria", f"{excellent_count}/10")
            
            # Save form
            st.write("---")
            
            with st.form("save_sbox_form"):
                sbox_name = st.text_input(
                    "S-box Name:",
                    value=f"S-box_{len(manager.list_sboxes()) + 1}",
                    help="Enter a unique name for this S-box"
                )
                
                description = st.text_area(
                    "Description (optional):",
                    help="Add notes about this S-box"
                )
                
                # Get metadata if available
                metadata = {}
                if hasattr(st.session_state, 'affine_matrix'):
                    metadata['affine_matrix'] = st.session_state.affine_matrix.tolist()
                if hasattr(st.session_state, 'constant'):
                    metadata['constant'] = st.session_state.constant.tolist()
                if description:
                    metadata['description'] = description
                
                submit = st.form_submit_button("💾 Save S-box", width="stretch")
                
                if submit:
                    if sbox_name in manager.list_sboxes():
                        st.error(f"❌ S-box named '{sbox_name}' already exists!")
                    else:
                        manager.save_sbox(
                            sbox_name,
                            st.session_state.constructed_sbox,
                            st.session_state.test_results,
                            metadata
                        )
                        st.success(f"✅ S-box '{sbox_name}' saved successfully!")
                        st.balloons()
                        st.rerun()
    
    # Tab 2: Saved S-boxes
    with tab2:
        st.subheader("Saved S-boxes")
        
        saved_list = manager.list_sboxes()
        
        if not saved_list:
            st.info("📭 No saved S-boxes yet. Save your first S-box in the 'Save Current S-box' tab.")
        else:
            st.success(f"📦 {len(saved_list)} S-box(es) saved")
            
            # Display saved S-boxes
            for name in saved_list:
                with st.expander(f"🔍 {name}", expanded=False):
                    sbox_data = manager.get_sbox(name)
                    
                    # Full S-box display
                    st.write("**S-box Table:**")
                    sbox_df = pd.DataFrame(
                        sbox_data['sbox'],
                        columns=[f"{i:X}" for i in range(16)],
                        index=[f"{i:X}" for i in range(16)]
                    )
                    st.dataframe(sbox_df, width="stretch")
                    
                    st.write("---")
                    
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.write("**Test Results:**")
                        results = sbox_data['results']
                        
                        results_df = pd.DataFrame({
                            'Metric': ['NL', 'SAC', 'BIC-NL', 'BIC-SAC', 'LAP', 'DAP', 
                                     'DU', 'AD', 'TO', 'CI'],
                            'Value': [
                                float(results["NL"]["value"]),
                                float(results["SAC"]["value"]),
                                float(results["BIC-NL"]["value"]),
                                float(results["BIC-SAC"]["value"]),
                                float(results["LAP"]["value"]),
                                float(results["DAP"]["value"]),
                                float(results["DU"]["value"]),
                                float(results["AD"]["value"]),
                                float(results["TO"]["value"]),
                                float(results["CI"]["value"]),
                            ]
                        })
                        st.dataframe(results_df, hide_index=True, width="stretch")
                        
                        # Calculate SV
                        sv = (120 - results['NL']['value']) + \
                             abs(0.5 - results['SAC']['value']) + \
                             (120 - results['BIC-NL']['value']) + \
                             abs(0.5 - results['BIC-SAC']['value'])
                        st.metric("Strength Value", f"{sv:.6f}")
                    
                    with col2:
                        st.write("**Metadata:**")
                        st.text(f"Saved: {sbox_data['timestamp']}")
                        
                        if 'description' in sbox_data['metadata']:
                            st.text_area(
                                "Description:",
                                value=sbox_data['metadata']['description'],
                                disabled=True,
                                key=f"desc_{name}",
                                height=100
                            )
                        
                        # Show affine matrix info if available
                        if 'affine_matrix' in sbox_data['metadata']:
                            affine_first_row = sbox_data['metadata']['affine_matrix'][0]
                            first_row_str = ''.join(map(str, affine_first_row))
                            first_row_dec = int(first_row_str, 2)
                            st.info(f"**Affine Matrix First Row:**\n\nBinary: {first_row_str}\n\nDecimal: {first_row_dec}")
                        
                        if 'constant' in sbox_data['metadata']:
                            constant = sbox_data['metadata']['constant']
                            const_str = ''.join(map(str, constant))
                            const_dec = int(const_str, 2)
                            st.info(f"**8-bit Constant:**\n\nBinary: {const_str}\n\nDecimal: {const_dec}")
                    
                    with col3:
                        st.write("**Actions:**")
                        
                        if st.button("🔄 Load", key=f"load_{name}", width="stretch"):
                            st.session_state.constructed_sbox = sbox_data['sbox']
                            st.session_state.test_results = sbox_data['results']
                            st.session_state.sbox_name = name
                            st.success(f"✅ Loaded '{name}'")
                            st.rerun()
                        
                        if st.button("📥 Export", key=f"export_{name}", width="stretch"):
                            # Create export data
                            export_text = f"""S-box: {name}
Timestamp: {sbox_data['timestamp']}

Test Results:
=============
NL: {results['NL']['value']}
SAC: {results['SAC']['value']:.6f}
BIC-NL: {results['BIC-NL']['value']}
BIC-SAC: {results['BIC-SAC']['value']:.6f}
LAP: {results['LAP']['value']:.6f}
DAP: {results['DAP']['value']:.6f}
DU: {results['DU']['value']}
AD: {results['AD']['value']}
TO: {results['TO']['value']:.6f}
CI: {results['CI']['value']}

Strength Value: {sv:.6f}

S-box Table:
============
"""
                            # Add S-box in hex format
                            for i in range(16):
                                # row_hex = ' '.join(f"{val:02X}" for val in sbox_data['sbox'][i])
                                row = ' '.join(str(val) for val in sbox_data['sbox'][i])
                                export_text += f"{row}\n"
                            
                            st.download_button(
                                "Download",
                                data=export_text,
                                file_name=f"{name}.txt",
                                mime="text/plain",
                                key=f"dl_{name}"
                            )
                        
                        if st.button("🗑️ Delete", key=f"del_{name}", width="stretch"):
                            manager.delete_sbox(name)
                            st.success(f"✅ Deleted '{name}'")
                            st.rerun()
    
    # Tab 3: Side-by-Side Comparison
    with tab3:
        st.subheader("Side-by-Side Comparison")
        
        saved_list = manager.list_sboxes()
        
        if len(saved_list) < 2:
            st.info("ℹ️ Save at least 2 S-boxes to compare them.")
        else:
            st.write("Select S-boxes to compare:")
            
            # Multi-select for comparison
            selected = st.multiselect(
                "Choose S-boxes:",
                options=saved_list,
                default=saved_list[:min(3, len(saved_list))],
                max_selections=5,
                help="Select 2-5 S-boxes to compare"
            )
            
            if len(selected) >= 2:
                # Comparison table
                comparison_df = manager.compare_sboxes(selected)
                
                st.write("### Comparison Table")
                
                # Highlight best values
                def highlight_best(row):
                    metric = row['Metric']
                    colors = [''] * len(row)
                    
                    # Skip metric name column
                    values = row[1:].values
                    
                    # Determine best based on metric
                    if metric in ['NL', 'BIC-NL', 'AD', 'CI']:
                        best_idx = np.argmax(values) + 1
                    elif metric in ['SAC', 'BIC-SAC']:
                        # Closest to 0.5
                        best_idx = np.argmin(np.abs(values - 0.5)) + 1
                    elif metric in ['LAP', 'DAP', 'DU', 'TO']:
                        best_idx = np.argmin(values) + 1
                    else:
                        best_idx = -1
                    
                    if best_idx > 0:
                        colors[best_idx] = 'background-color: #90EE90'
                    
                    return colors
                
                styled_df = comparison_df.style.apply(highlight_best, axis=1)
                st.dataframe(styled_df, width="stretch", hide_index=True)
                
                st.info("💡 **Green** = Best value for that metric")
                
                # Visual comparison
                st.write("### Visual Comparison")
                
                viz_type = st.radio(
                    "Visualization type:",
                    ["Radar Chart", "Bar Chart", "Heatmap"],
                    horizontal=True
                )
                
                if viz_type == "Radar Chart":
                    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
                    
                    metrics = ['NL', 'SAC', 'BIC-NL', 'BIC-SAC', 'LAP', 'DAP', 
                              'DU', 'AD', 'TO', 'CI']
                    
                    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
                    angles += angles[:1]
                    
                    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
                    
                    for i, name in enumerate(selected):
                        sbox_data = manager.get_sbox(name)
                        results = sbox_data['results']
                        
                        # Normalize values
                        values = [
                            results['NL']['value'] / 112.0,
                            1 - abs(results['SAC']['value'] - 0.5) * 2,
                            results['BIC-NL']['value'] / 112.0,
                            1 - abs(results['BIC-SAC']['value'] - 0.5) * 2,
                            1 - (results['LAP']['value'] / 0.125),
                            1 - (results['DAP']['value'] / 0.03125),
                            1 - (results['DU']['value'] / 8.0),
                            results['AD']['value'] / 7.0,
                            1 - min(results['TO']['value'], 1.0),
                            min(results['CI']['value'] / 5.0, 1.0)
                        ]
                        
                        values = [max(0, min(1, v)) for v in values]
                        values += values[:1]
                        
                        ax.plot(angles, values, 'o-', linewidth=2, 
                               label=name, color=colors[i % len(colors)])
                        ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
                    
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(metrics)
                    ax.set_ylim(0, 1)
                    ax.grid(True)
                    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
                    
                    st.pyplot(fig)
                    plt.close()
                
                elif viz_type == "Bar Chart":
                    # Create grouped bar chart
                    metrics = ['NL', 'SAC', 'BIC-NL', 'BIC-SAC', 'LAP', 'DAP', 
                              'DU', 'AD', 'TO', 'CI']
                    
                    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
                    axes = axes.flatten()
                    
                    for i, metric in enumerate(metrics):
                        ax = axes[i]
                        values = []
                        
                        for name in selected:
                            sbox_data = manager.get_sbox(name)
                            values.append(sbox_data['results'][metric]['value'])
                        
                        ax.bar(range(len(selected)), values, color='#2E86AB')
                        ax.set_xticks(range(len(selected)))
                        ax.set_xticklabels(selected, rotation=45, ha='right')
                        ax.set_title(metric, fontweight='bold')
                        ax.grid(axis='y', alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                elif viz_type == "Heatmap":
                    # Create heatmap
                    data = []
                    for name in selected:
                        sbox_data = manager.get_sbox(name)
                        results = sbox_data['results']
                        data.append([
                            results['NL']['value'],
                            results['SAC']['value'],
                            results['BIC-NL']['value'],
                            results['BIC-SAC']['value'],
                            results['LAP']['value'],
                            results['DAP']['value'],
                            results['DU']['value'],
                            results['AD']['value'],
                            results['TO']['value'],
                            results['CI']['value']
                        ])
                    
                    fig, ax = plt.subplots(figsize=(12, len(selected) * 1.5))
                    
                    # Normalize for visualization
                    data_array = np.array(data)
                    data_norm = (data_array - data_array.min(axis=0)) / \
                               (data_array.max(axis=0) - data_array.min(axis=0) + 1e-10)
                    
                    sns.heatmap(data_norm, annot=data_array, fmt='.3f', cmap='RdYlGn',
                               xticklabels=['NL', 'SAC', 'BIC-NL', 'BIC-SAC', 'LAP', 
                                          'DAP', 'DU', 'AD', 'TO', 'CI'],
                               yticklabels=selected, ax=ax, cbar_kws={'label': 'Normalized'})
                    ax.set_title('S-box Comparison Heatmap')
                    
                    st.pyplot(fig)
                    plt.close()
            
            else:
                st.warning("⚠️ Please select at least 2 S-boxes to compare.")
    
    # Tab 4: Rankings
    with tab4:
        st.subheader("S-box Rankings")
        
        ranking_df = manager.calculate_ranking()
        
        if ranking_df.empty:
            st.info("📭 No saved S-boxes to rank.")
        else:
            st.write(f"### Rankings ({len(ranking_df)} S-boxes)")
            
            # Display ranking table
            st.dataframe(
                ranking_df.style.background_gradient(subset=['SV'], cmap='RdYlGn_r'),
                width="stretch",
                hide_index=True
            )
            
            st.info("💡 **Lower SV** = Stronger S-box. Green = Best, Red = Worst")
            
            # Top 3 podium
            if len(ranking_df) >= 3:
                st.write("### 🏆 Top 3 S-boxes")
                
                col1, col2, col3 = st.columns(3)
                
                for i, col in enumerate([col2, col1, col3]):  # 2nd, 1st, 3rd
                    rank_idx = [1, 0, 2][i]
                    if rank_idx < len(ranking_df):
                        row = ranking_df.iloc[rank_idx]
                        
                        with col:
                            medal = ['🥇', '🥈', '🥉'][rank_idx]
                            st.markdown(f"### {medal} #{row['Rank']}")
                            st.markdown(f"**{row['S-box Name']}**")
                            st.metric("SV", f"{row['SV']:.6f}")
                            st.metric("Criteria", row['Excellent Criteria'])
    
    # Tab 5: Statistical Analysis
    with tab5:
        st.subheader("Statistical Analysis")
        
        saved_list = manager.list_sboxes()
        
        if len(saved_list) < 2:
            st.info("ℹ️ Save at least 2 S-boxes for statistical analysis.")
        else:
            st.write(f"Analyzing {len(saved_list)} S-boxes...")
            
            # Collect all data
            all_data = {
                'NL': [], 'SAC': [], 'BIC-NL': [], 'BIC-SAC': [],
                'LAP': [], 'DAP': [], 'DU': [], 'AD': [], 'TO': [], 'CI': []
            }
            
            for name in saved_list:
                sbox_data = manager.get_sbox(name)
                results = sbox_data['results']
                
                for metric in all_data.keys():
                    all_data[metric].append(results[metric]['value'])
            
            # Statistical summary
            st.write("### Statistical Summary")
            
            stats_data = []
            for metric, values in all_data.items():
                stats_data.append({
                    'Metric': metric,
                    'Mean': np.mean(values),
                    'Median': np.median(values),
                    'Std Dev': np.std(values),
                    'Min': np.min(values),
                    'Max': np.max(values)
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(
                stats_df.style.format({
                    'Mean': '{:.4f}',
                    'Median': '{:.4f}',
                    'Std Dev': '{:.4f}',
                    'Min': '{:.4f}',
                    'Max': '{:.4f}'
                }),
                width="stretch",
                hide_index=True
            )
            
            # Distribution plots
            st.write("### Metric Distributions")
            
            metric_choice = st.selectbox(
                "Select metric to visualize:",
                options=list(all_data.keys())
            )
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Histogram
            ax1.hist(all_data[metric_choice], bins=10, color='#2E86AB', edgecolor='black', alpha=0.7)
            ax1.set_xlabel(metric_choice)
            ax1.set_ylabel('Frequency')
            ax1.set_title(f'{metric_choice} Distribution')
            ax1.grid(axis='y', alpha=0.3)
            
            # Box plot
            ax2.boxplot(all_data[metric_choice], vert=True)
            ax2.set_ylabel(metric_choice)
            ax2.set_title(f'{metric_choice} Box Plot')
            ax2.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Correlation matrix
            st.write("### Metric Correlation Matrix")
            
            corr_data = pd.DataFrame(all_data)
            corr_matrix = corr_data.corr()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, vmin=-1, vmax=1, ax=ax,
                       cbar_kws={'label': 'Correlation'})
            ax.set_title('Metric Correlation Matrix')
            
            st.pyplot(fig)
            plt.close()
            
            st.info("💡 **Interpretation:** +1 = perfect positive correlation, -1 = perfect negative correlation, 0 = no correlation")


# Main execution for standalone testing
if __name__ == "__main__":
    st.set_page_config(
        page_title="Results & Comparison",
        page_icon="📊",
        layout="wide"
    )
    
    render_results_comparison()
