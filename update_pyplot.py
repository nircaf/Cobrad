import re

with open('6_hep_group_comparison.py', 'r') as f:
    text = f.read()

old_func = '''def log_and_plot_fig(fig, title=None, description=None, use_container_width=True):
    """
    Plots a figure to Streamlit and logs it in session state for PPTX export.
    """
    if 'pptx_figures_data' not in st.session_state:
        st.session_state.pptx_figures_data = []

    st.pyplot(fig, use_container_width=use_container_width)

    if PPTX_AVAILABLE:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        desc_str = description if description else (title if title else "Figure")
        st.session_state.pptx_figures_data.append({
            'title': title if title else "Figure",
            'description': desc_str,
            'image': buf
        })'''

new_func = '''_original_st_pyplot = st.pyplot

def custom_st_pyplot(fig=None, clear_figure=None, **kwargs):
    if fig is None:
        fig = plt.gcf()
        
    title = "Figure"
    try:
        if fig._suptitle:
            title = fig._suptitle.get_text()
        else:
            for ax in fig.axes:
                if ax.get_title():
                    title = ax.get_title()
                    break
    except:
        pass
        
    description = f"Plot: {title}\\n"
    
    try:
        stats = []
        for ax in fig.axes:
            lines = ax.get_lines()
            for line in lines:
                ydata = line.get_ydata()
                if len(ydata) > 0:
                    # check if ydata is numeric
                    if hasattr(ydata, 'dtype') and np.issubdtype(ydata.dtype, np.number):
                        label = line.get_label()
                        stat_line = f"  {label if label and not label.startswith('_') else 'Data Line'}: Min={np.nanmin(ydata):.3f}, Max={np.nanmax(ydata):.3f}, Mean={np.nanmean(ydata):.3f}"
                        if stat_line not in stats:
                            stats.append(stat_line)
        if stats:
            description += "Numerical Summary:\\n" + "\\n".join(stats)
            
        # extract histogram data if present
        hist_stats = []
        for ax in fig.axes:
            patches = ax.patches
            if len(patches) > 5: # likely a histogram
                heights = [p.get_height() for p in patches if isinstance(p, plt.Rectangle)]
                if heights:
                    hist_stats.append(f"  Histogram: {sum(heights):.0f} total items, max bin freq={max(heights):.0f}")
        if hist_stats:
            description += "\\n" + "\\n".join(hist_stats)
    except:
        pass
        

    if 'pptx_figures_data' not in st.session_state:
        st.session_state.pptx_figures_data = []

    _original_st_pyplot(fig, clear_figure=clear_figure, **kwargs)

    if PPTX_AVAILABLE:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        st.session_state.pptx_figures_data.append({
            'title': title,
            'description': description,
            'image': buf
        })

st.pyplot = custom_st_pyplot'''

text = text.replace(old_func, new_func)

with open('6_hep_group_comparison.py', 'w') as f:
    f.write(text)

print("Replaced!")
