import streamlit as st
import os
import subprocess
import shlex
import json
import pandas as pd
import time

def _load_saved_projects(config_path='uploaded_projects.txt'):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            projects = [line.strip() for line in f if line.strip()]
    else:
        projects = []
    return projects

def _save_project_name(project_name, config_path='uploaded_projects.txt'):
    projects = _load_saved_projects(config_path)
    if project_name not in projects:
        projects.append(project_name)
        with open(config_path, 'w') as f:
            for p in projects:
                f.write(p + '\n')

def _save_uploaded_files(uploaded_files, project_name, clinical_table_bytes=None, clinical_table_name=None, key_col=None):
    """
    Save uploaded EEG/EDF files to EDF_Format/{project_name} and save the clinical table there as well.
    Returns the destination folder path.
    """
    import os
    os.makedirs(f'EDF_Format/{project_name}', exist_ok=True)
    saved = []
    for up in uploaded_files:
        try:
            dest = os.path.join('EDF_Format', project_name, up.name)
            with open(dest, 'wb') as f:
                f.write(up.getbuffer())
            saved.append(dest)
        except Exception as e:
            st.warning(f'Could not save {up.name}: {e}')
    # Save clinical table bytes if provided
    if clinical_table_bytes is not None and clinical_table_name is not None:
        try:
            dest = os.path.join('EDF_Format', project_name, clinical_table_name)
            with open(dest, 'wb') as f:
                f.write(clinical_table_bytes)
        except Exception as e:
            st.warning(f'Could not save clinical table {clinical_table_name}: {e}')
    # Save metadata for later merging
    try:
        meta = {
            'clinical_table': clinical_table_name,
            'clinical_key_col': key_col,
            'saved_files': saved
        }
        with open(os.path.join('EDF_Format', project_name, '_upload_meta.json'), 'w') as f:
            json.dump(meta, f)
    except Exception as e:
        st.warning(f'Could not write project metadata: {e}')
    return os.path.join('EDF_Format', project_name)

def _run_cleaning_pipeline(project_name):
    """
    Launch the cleaning pipeline in background using a shell call and log output to EDF_Format/{project}/pipeline.log
    """
    cmd = f'python3 1_edf_cleaning.py -c {shlex.quote(project_name)}'
    log_dir = os.path.join('EDF_Format', project_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'pipeline.log')
    try:
        with open(log_file, 'ab') as f:
            subprocess.Popen(cmd, shell=True, stdout=f, stderr=f)
        return True
    except Exception as e:
        st.error(f'Failed to start cleaning pipeline: {e}')
        return False

def run_upload_page():
    st.title('Upload or Modify Project')
    st.info('Upload EEG/EDF files and a clinical table (.csv/.xlsx). You can create a new project or modify an existing one.')

    mode = st.radio('Mode', ['Create new project', 'Modify existing project'])

    # discover projects from saved list and EDF_Format folders
    saved = _load_saved_projects()
    edf_root = 'EDF_Format'
    folders = []
    if os.path.isdir(edf_root):
        folders = [d for d in os.listdir(edf_root) if os.path.isdir(os.path.join(edf_root, d))]
    all_projects = list(dict.fromkeys(saved + folders))

    if mode == 'Create new project':
        new_project_name = st.text_input('Project name (no spaces):', value='MyProject')
        uploaded_files = st.file_uploader('Upload EEG/EDF files (multiple)', accept_multiple_files=True, type=['edf', 'EEG', 'eeg'], key='upload_files_page')
        clinical_table = st.file_uploader('Upload clinical table (.csv or .xlsx)', accept_multiple_files=False, type=['csv', 'xls', 'xlsx'], key='clinical_table_page')
        clinical_key_col = None
        if clinical_table is not None:
            try:
                if clinical_table.name.endswith('.csv'):
                    clinical_df = pd.read_csv(clinical_table)
                else:
                    clinical_df = pd.read_excel(clinical_table)
                st.write('Clinical table preview:')
                st.dataframe(clinical_df.head())
                cols = list(clinical_df.columns)
                clinical_key_col = st.selectbox('Select column that contains file or folder name', cols)
            except Exception as e:
                st.error(f'Could not read clinical table: {e}')
        start_processing = st.button('Create & Start Processing')
        if start_processing:
            if not new_project_name:
                st.error('Please provide a project name')
            elif not uploaded_files or len(uploaded_files) == 0:
                st.error('Please upload at least one EEG/EDF file')
            elif clinical_table is None:
                st.error('Please upload a clinical table file')
            elif clinical_key_col is None:
                st.error('Please select the clinical key column')
            else:
                clinical_bytes = clinical_table.getvalue()
                clinical_name = clinical_table.name
                dest_folder = _save_uploaded_files(uploaded_files, new_project_name, clinical_table_bytes=clinical_bytes, clinical_table_name=clinical_name, key_col=clinical_key_col)
                _save_project_name(new_project_name)
                st.success(f'Files saved to {dest_folder}. Starting cleaning pipeline...')
                ok = _run_cleaning_pipeline(new_project_name)
                if ok:
                    st.success('Cleaning pipeline started in background. Check logs in terminal or EDF_Format/{project}/pipeline.log')
                else:
                    st.error('Failed to start cleaning pipeline.')

    else:  # Modify existing
        if not all_projects:
            st.info('No projects found. Create one first.')
            return
        project_name = st.selectbox('Select project to modify', all_projects)
        project_dir = os.path.join(edf_root, project_name)
        os.makedirs(project_dir, exist_ok=True)
        st.write(f'Files in project `{project_name}`:')
        files = sorted(os.listdir(project_dir))
        # show files with checkboxes to delete
        to_delete = st.multiselect('Select files to delete', files)
        if st.button('Delete selected files'):
            for fn in to_delete:
                try:
                    os.remove(os.path.join(project_dir, fn))
                except Exception as e:
                    st.warning(f'Could not delete {fn}: {e}')
            st.experimental_rerun()

        st.write('Add more files to project:')
        added_files = st.file_uploader('Upload additional EEG/EDF files (multiple)', accept_multiple_files=True, type=['edf', 'EEG', 'eeg'], key=f'add_files_{project_name}')
        added_clinical = st.file_uploader('(Optional) Upload / replace clinical table (.csv/.xlsx)', accept_multiple_files=False, type=['csv', 'xls', 'xlsx'], key=f'add_clinical_{project_name}')
        add_key_col = None
        if added_clinical is not None:
            try:
                if added_clinical.name.endswith('.csv'):
                    preview_df = pd.read_csv(added_clinical)
                else:
                    preview_df = pd.read_excel(added_clinical)
                st.write('Clinical table preview:')
                st.dataframe(preview_df.head())
                add_key_col = st.selectbox('Select clinical key column', list(preview_df.columns), key=f'keycol_{project_name}')
            except Exception as e:
                st.error(f'Could not read clinical table: {e}')

        if st.button('Apply changes (save uploaded files)'):
            # save added files
            saved = []
            if added_files:
                for up in added_files:
                    try:
                        dest = os.path.join(project_dir, up.name)
                        with open(dest, 'wb') as f:
                            f.write(up.getbuffer())
                        saved.append(dest)
                    except Exception as e:
                        st.warning(f'Could not save {up.name}: {e}')
            if added_clinical is not None:
                try:
                    dest = os.path.join(project_dir, added_clinical.name)
                    with open(dest, 'wb') as f:
                        f.write(added_clinical.getbuffer())
                    # update meta
                    meta_path = os.path.join(project_dir, '_upload_meta.json')
                    meta = {}
                    if os.path.exists(meta_path):
                        try:
                            with open(meta_path, 'r') as mf:
                                meta = json.load(mf)
                        except Exception:
                            meta = {}
                    meta.update({'clinical_table': added_clinical.name, 'clinical_key_col': add_key_col})
                    with open(meta_path, 'w') as mf:
                        json.dump(meta, mf)
                except Exception as e:
                    st.warning(f'Could not save clinical table: {e}')
            st.success('Changes applied.')

        # Run cleaning and wait
        if st.button('Run cleaning pipeline and wait (blocking)'):
            cmd = f'python3 1_edf_cleaning.py -c {shlex.quote(project_name)}'
            log_file = os.path.join(project_dir, 'pipeline.log')
            st.info(f'Running: {cmd} — output will be saved to {log_file}')
            with st.spinner('Cleaning pipeline running — this will block until finished'):
                try:
                    # run and capture output
                    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    out = proc.stdout
                    # write to log
                    with open(log_file, 'a') as lf:
                        lf.write(out)
                    st.subheader('Pipeline output')
                    st.text(out)
                    if proc.returncode == 0:
                        st.success('Cleaning pipeline finished successfully')
                    else:
                        st.error(f'Cleaning pipeline exited with code {proc.returncode}')
                except Exception as e:
                    st.error(f'Error running pipeline: {e}')
