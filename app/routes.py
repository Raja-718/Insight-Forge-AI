from flask import Blueprint, render_template, request, jsonify, session, send_file
from agents.orchestrator import Orchestrator
from src.data_loader import load_file
from src import pma_engine as pma
from src.bia import bia_engine as bia
from agents.ml_agent import MLAgent
import os, json, pickle
import pandas as pd
import numpy as np

_ml_agent = MLAgent()

main = Blueprint('main', __name__)
orchestrator = Orchestrator()

# ── keyword map (same as before) ─────────────────────────────
KEYWORD_MAP = {
    'value':    ['sales','amount','revenue','price','total','income','value','profit','earn','turnover','gross','net','cost','expense','budget','salary','wage','fee','marks','score','grade','gpa','percentage','percent','rating','points','likes','followers','views','clicks','impressions','reach','engagement','cases','deaths','patients','count','number','qty','quantity','units','attendance','hours','duration','age','weight','height','temperature','population','votes','donations','fund','investment','return','loss','orders','tickets','calls','downloads','installs','purchases','transactions'],
    'qty':      ['qty','quantity','units','count','volume','num','sold','orders','items','pieces','students','employees','patients','users','members','followers','likes','views','clicks','cases','records','entries','rows','responses'],
    'date':     ['date','time','day','month','year','period','week','quarter','timestamp','created','updated','joined','enrolled','admitted','posted','published','submitted','recorded','reported','born','dob','start','end','deadline'],
    'category': ['category','cat','type','product','class','group','segment','kind','line','brand','dept','department','division','subject','course','stream','major','field','sector','industry','domain','topic','tag','label','status','stage','gender','grade','level','rank','tier','plan','package','model','series','platform','channel','source','medium','campaign','post_type','content_type'],
    'region':   ['region','area','zone','location','city','state','country','territory','district','branch','market','store','campus','school','college','hospital','office','site','address','place','village','town','province','county','ward','block','cluster'],
    'person':   ['rep','agent','employee','staff','person','salesperson','seller','assigned','owner','handler','manager','teacher','professor','doctor','nurse','student','user','author','creator','by','name','handled_by','assigned_to','reported_by'],
    'channel':  ['channel','platform','source','medium','store','outlet','mode','method','via','network','social','site','app','device','browser','os','carrier'],
    'segment':  ['customer_type','customer','client','buyer','segment','tier','membership','student_type','employee_type','patient_type','user_type','account_type','subscription','plan'],
    'rate':     ['discount','disc','rebate','reduction','off','rate','ratio','pct','percent','percentage','tax','commission','margin','growth','change','pass_rate','fail_rate','attendance_rate','conversion','ctr','bounce'],
}

def detect_col(columns, key):
    keywords = KEYWORD_MAP.get(key, [])
    for col in columns:
        if any(k in col.lower().replace(' ','_') for k in keywords):
            return col
    return None

def smart_detect_all(df):
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    value_col = detect_col(num_cols,'value') or (num_cols[0] if num_cols else None)
    qty_col   = detect_col(num_cols,'qty')
    if qty_col == value_col: qty_col = next((c for c in num_cols if c!=value_col),None)
    rate_col  = detect_col(num_cols,'rate')
    date_col  = detect_col(df.columns.tolist(),'date')
    cat_col   = detect_col(cat_cols,'category') or (cat_cols[0] if cat_cols else None)
    region_col= detect_col(cat_cols,'region')
    person_col= detect_col(cat_cols,'person')
    channel_col=detect_col(cat_cols,'channel')
    segment_col=detect_col(cat_cols,'segment')
    cat2_col  = next((c for c in [segment_col,channel_col,person_col] if c and c!=cat_col),None)
    return {'value':value_col,'qty':qty_col,'rate':rate_col,'date':date_col,'category':cat_col,
            'region':region_col,'person':person_col,'channel':channel_col,'segment':segment_col,
            'cat2':cat2_col,'num_cols':num_cols,'cat_cols':cat_cols,'all_cols':df.columns.tolist()}

# ── Routes ────────────────────────────────────────────────────

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/analysis')
def analysis():
    return render_template('analysis.html')

@main.route('/predict')
def predict():
    return render_template('predict.html')

@main.route('/bi')
def bi():
    return render_template('bi.html')

@main.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@main.route('/upload')
def upload():
    return render_template('upload.html')

@main.route('/chat')
def chat():
    return render_template('chat.html')

# ── New Data Analysis routes ───────────────────────────────────
@main.route('/data-preview')
def data_preview():
    return render_template('data_preview.html')

@main.route('/analysis-dashboard')
def analysis_dashboard():
    return render_template('analysis_dashboard.html')

@main.route('/auto-dashboard')
def auto_dashboard():
    return render_template('auto_dashboard.html')

# ── API Endpoints ──────────────────────────────────────────────

@main.route('/api/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if not file: return jsonify({'error':'No file provided'}),400
    os.makedirs('uploads',exist_ok=True)
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)
    session['uploaded_file'] = filepath
    return jsonify({'message':'File uploaded successfully','path':filepath,'filename':file.filename})

@main.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.get_json()
    user_message = data.get('message','').strip()
    if not user_message: return jsonify({'error':'No message provided'}),400
    file_path = session.get('uploaded_file') or data.get('file_path')
    response = orchestrator.run(user_message, file_path)
    return jsonify({'response': response})

@main.route('/api/session-file', methods=['GET'])
def get_session_file():
    return jsonify({'file_path': session.get('uploaded_file',None)})

# ── NEW: Preview data endpoint ─────────────────────────────────
@main.route('/api/preview-data', methods=['POST'])
def preview_data():
    data = request.get_json()
    file_path = session.get('uploaded_file') or data.get('file_path')
    if not file_path: return jsonify({'error':'No file in session'}),400
    try:
        df = load_file(file_path)
        df.columns = [str(c).strip() for c in df.columns]

        # Build profile
        num_cols = df.select_dtypes(include='number').columns.tolist()
        total_cells = df.shape[0] * df.shape[1]
        total_missing = int(df.isnull().sum().sum())
        completeness = round((1 - total_missing/total_cells)*100, 1) if total_cells > 0 else 100
        duplicates = int(df.duplicated().sum())

        col_profiles = []
        for col in df.columns:
            miss_count = int(df[col].isnull().sum())
            miss_pct   = round(miss_count / len(df) * 100, 1)
            is_num = col in num_cols
            p = {
                'name': col,
                'dtype': 'number' if is_num else 'text',
                'missing_count': miss_count,
                'missing_pct': miss_pct,
                'unique': int(df[col].nunique()),
            }
            if is_num:
                p.update({
                    'min':    round(float(df[col].min()),2)    if not df[col].isnull().all() else None,
                    'max':    round(float(df[col].max()),2)    if not df[col].isnull().all() else None,
                    'mean':   round(float(df[col].mean()),2)   if not df[col].isnull().all() else None,
                    'median': round(float(df[col].median()),2) if not df[col].isnull().all() else None,
                    'std':    round(float(df[col].std()),2)    if not df[col].isnull().all() else None,
                })
            else:
                top_val = df[col].value_counts().index[0] if not df[col].isnull().all() and len(df[col].dropna()) > 0 else None
                p['top'] = str(top_val) if top_val is not None else None
            col_profiles.append(p)

        profile = {
            'completeness': completeness,
            'total_missing': total_missing,
            'duplicates': duplicates,
            'columns': col_profiles,
        }

        # Convert rows to JSON-safe format
        df_safe = df.where(pd.notnull(df), None)
        # Limit to 500 rows for performance; full data available via pagination
        rows = df_safe.head(500).to_dict(orient='records')
        # Clean NaN/inf values
        for row in rows:
            for k,v in row.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    row[k] = None

        return jsonify({
            'columns': df.columns.tolist(),
            'rows':    rows,
            'total_rows': len(df),
            'profile': profile,
        })
    except Exception as e:
        return jsonify({'error': str(e)}),500

# ── NEW: Edit data endpoint ────────────────────────────────────
@main.route('/api/edit-data', methods=['POST'])
def edit_data():
    data = request.get_json()
    file_path = session.get('uploaded_file') or data.get('file_path')
    if not file_path: return jsonify({'error':'No file in session'}),400
    action = data.get('action')
    column = data.get('column','')
    value  = data.get('value','')
    try:
        df = load_file(file_path)
        df.columns = [str(c).strip() for c in df.columns]

        if action == 'rename':
            if not value: return jsonify({'error':'New name is required'}),400
            df = df.rename(columns={column: value})
            msg = f'Column "{column}" renamed to "{value}"'

        elif action == 'drop-col':
            if column not in df.columns: return jsonify({'error':f'Column "{column}" not found'}),400
            df = df.drop(columns=[column])
            msg = f'Column "{column}" dropped'

        elif action == 'fill-missing':
            if column not in df.columns: return jsonify({'error':f'Column "{column}" not found'}),400
            if value == 'mean':   df[column].fillna(df[column].mean(), inplace=True)
            elif value == 'median': df[column].fillna(df[column].median(), inplace=True)
            elif value == 'mode': df[column].fillna(df[column].mode()[0], inplace=True)
            elif value == '0':    df[column].fillna(0, inplace=True)
            else: df[column].fillna(value, inplace=True)
            msg = f'Missing values in "{column}" filled with {value}'

        elif action == 'drop-duplicates':
            before = len(df)
            df = df.drop_duplicates()
            msg = f'Removed {before - len(df)} duplicate rows'

        elif action == 'sort':
            if column not in df.columns: return jsonify({'error':f'Column "{column}" not found'}),400
            df = df.sort_values(by=column, ascending=(value=='asc'))
            msg = f'Sorted by "{column}" {"ascending" if value=="asc" else "descending"}'

        elif action == 'filter-rows':
            before = len(df)
            try:
                df = df.query(f'`{column}` {value}')
                msg = f'Filtered: kept {len(df)} of {before} rows'
            except Exception as fe:
                return jsonify({'error': f'Filter error: {str(fe)}'}),400
        else:
            return jsonify({'error':'Unknown action'}),400

        # Save back
        if file_path.endswith('.csv'):
            df.to_csv(file_path, index=False)
        else:
            df.to_excel(file_path, index=False)

        return jsonify({'message': msg, 'rows': len(df), 'columns': len(df.columns)})
    except Exception as e:
        return jsonify({'error': str(e)}),500

# ── Chart data (existing) ──────────────────────────────────────
@main.route('/api/chart-data', methods=['POST'])
def chart_data():
    data = request.get_json()
    file_path = session.get('uploaded_file') or data.get('file_path')
    if not file_path: return jsonify({'error':'No file in session.'}),400
    try:
        df = load_file(file_path)
        df.columns = [str(c).strip() for c in df.columns]
        cols = smart_detect_all(df)
        result = {'detected_columns':{k:v for k,v in cols.items() if v and k not in ['num_cols','cat_cols','all_cols']}}
        value_col=cols['value']; qty_col=cols['qty']; rate_col=cols['rate']
        date_col=cols['date']; cat_col=cols['category']; region_col=cols['region']
        person_col=cols['person']; num_cols=cols['num_cols']

        if date_col and value_col:
            try:
                df[date_col]=pd.to_datetime(df[date_col],errors='coerce')
                df_t=df.dropna(subset=[date_col]).copy()
                dr=(df_t[date_col].max()-df_t[date_col].min()).days
                freq='Y' if dr>730 else ('M' if dr>60 else 'W')
                m=df_t.set_index(date_col)[value_col].resample(freq).sum().reset_index()
                fmt='%Y' if freq=='Y' else ('%Y-%m' if freq=='M' else '%Y-%m-%d')
                result['time_series']={'x':m[date_col].dt.strftime(fmt).tolist(),'y':m[value_col].round(2).tolist(),'x_label':date_col,'y_label':value_col}
            except: pass

        if cat_col and value_col:
            top=df.groupby(cat_col)[value_col].sum().sort_values(ascending=False).head(15).reset_index()
            result['bar_chart']={'x':top[cat_col].astype(str).tolist(),'y':top[value_col].round(2).tolist(),'x_label':cat_col,'y_label':value_col}

        if value_col:
            cv=df[value_col].dropna(); counts,edges=np.histogram(cv,bins=20)
            result['histogram']={'x':[round((edges[i]+edges[i+1])/2,2) for i in range(len(counts))],'y':counts.tolist(),'label':value_col}

        if cat_col and value_col:
            pie=df.groupby(cat_col)[value_col].sum().sort_values(ascending=False).head(8).reset_index()
            result['pie_chart']={'labels':pie[cat_col].astype(str).tolist(),'values':pie[value_col].round(2).tolist(),'label':f'{value_col} by {cat_col}'}

        x_col=qty_col or (num_cols[1] if len(num_cols)>1 else None)
        if x_col and value_col and x_col!=value_col:
            sc=df[[x_col,value_col]+([rate_col] if rate_col else [])+([cat_col] if cat_col else [])].dropna()
            result['scatter']={'x':sc[x_col].tolist(),'y':sc[value_col].round(2).tolist(),'color':sc[rate_col].tolist() if rate_col else [0]*len(sc),'labels':sc[cat_col].astype(str).tolist() if cat_col else ['']*len(sc),'x_label':x_col,'y_label':value_col,'color_label':rate_col or ''}

        if region_col and value_col:
            if person_col:
                agg=df.groupby([region_col,person_col])[value_col].sum().reset_index()
                ur=agg[region_col].unique().tolist(); rt=agg.groupby(region_col)[value_col].sum().to_dict()
                result['treemap']={'labels':['All']+ur+agg[person_col].astype(str).tolist(),'parents':['']+['All']*len(ur)+agg[region_col].tolist(),'values':[round(float(agg[value_col].sum()),2)]+[round(float(rt[r]),2) for r in ur]+agg[value_col].round(2).tolist()}
            else:
                r_df=df.groupby(region_col)[value_col].sum().reset_index()
                result['treemap']={'labels':['All']+r_df[region_col].astype(str).tolist(),'parents':['']+['All']*len(r_df),'values':[round(float(r_df[value_col].sum()),2)]+r_df[value_col].round(2).tolist()}

        if len(num_cols)>=3:
            hc=num_cols[:8]; corr=df[hc].corr().round(2)
            result['heatmap']={'x':hc,'y':hc,'z':corr.values.tolist()}

        if len(num_cols)>=2:
            bc=num_cols[:6]
            result['box_plot']={'cols':bc,'data':{c:df[c].dropna().tolist() for c in bc}}

        if value_col:
            result['summary']={'total':round(float(df[value_col].sum()),2),'records':len(df),'avg':round(float(df[value_col].mean()),2),'max':round(float(df[value_col].max()),2),'columns':len(df.columns),'value_label':value_col,'qty_total':int(df[qty_col].sum()) if qty_col else 0,'qty_label':qty_col or 'N/A'}

        return jsonify(result)
    except Exception as e:
        return jsonify({'error':str(e)}),500


# ══════════════════════════════════════════════════════════════
#  PMA — PREDICTIVE MODELING & ANALYSIS ROUTES
# ══════════════════════════════════════════════════════════════

# ── Step 1: Detect data type ──────────────────────────────────
@main.route('/api/pma/detect', methods=['POST'])
def pma_detect():
    data = request.get_json()
    file_path = session.get('uploaded_file') or data.get('file_path')
    target_col = data.get('target_col')
    if not file_path:
        return jsonify({'error': 'No file in session'}), 400
    try:
        df = load_file(file_path)
        df.columns = [str(c).strip() for c in df.columns]
        info = pma.detect_data_type(df, target_col)
        # Store in session
        session['pma_info'] = info
        session['pma_file'] = file_path
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Step 2: Get column list for target selection ──────────────
@main.route('/api/pma/columns', methods=['POST'])
def pma_columns():
    data = request.get_json()
    file_path = session.get('uploaded_file') or data.get('file_path')
    if not file_path:
        return jsonify({'error': 'No file in session'}), 400
    try:
        df = load_file(file_path)
        df.columns = [str(c).strip() for c in df.columns]
        num_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        sample = df.head(5).to_dict(orient='records')
        return jsonify({
            'all_cols': df.columns.tolist(),
            'num_cols': num_cols,
            'cat_cols': cat_cols,
            'n_rows': len(df),
            'sample': sample,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Step 3: Feature importance ────────────────────────────────
@main.route('/api/pma/feature-importance', methods=['POST'])
def pma_feature_importance():
    data = request.get_json()
    file_path = session.get('pma_file') or session.get('uploaded_file') or data.get('file_path')
    target_col = data.get('target_col') or session.get('pma_info', {}).get('target_col')
    if not file_path or not target_col:
        return jsonify({'error': 'Missing file or target column'}), 400
    try:
        df = load_file(file_path)
        df.columns = [str(c).strip() for c in df.columns]
        info = pma.detect_data_type(df, target_col)
        X, y, feature_names, encoders = pma.preprocess_tabular(df, target_col, info)
        importance = pma.get_feature_importance(X, y, feature_names, info['problem_type'])
        session['pma_info'] = info
        session['pma_target'] = target_col
        return jsonify({'features': importance, 'problem_type': info['problem_type'], 'data_type': info['data_type']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Step 4: Model catalog & recommendations ───────────────────
@main.route('/api/pma/models', methods=['POST'])
def pma_models():
    data = request.get_json()
    info = session.get('pma_info') or {}
    data_type = data.get('data_type') or info.get('data_type', 'tabular')
    problem_type = data.get('problem_type') or info.get('problem_type', 'classification')
    n_rows = data.get('n_rows') or info.get('n_rows', 1000)
    n_cols = data.get('n_cols') or info.get('n_cols', 10)
    try:
        catalog = pma.get_model_catalog(data_type, problem_type)
        recommendations = pma.recommend_models(data_type, problem_type, n_rows, n_cols)
        # AI explanation
        ai_note = ''
        try:
            ai_note = _ml_agent.explain_model_selection(
                {'data_type': data_type, 'problem_type': problem_type,
                 'n_rows': n_rows, 'n_cols': n_cols,
                 'target_col': info.get('target_col'),
                 'num_cols': info.get('num_cols', []),
                 'cat_cols': info.get('cat_cols', [])},
                recommendations
            )
        except Exception:
            ai_note = 'AI explanation unavailable.'
        models_list = [
            {'key': k, 'name': v['name'], 'category': v['category']}
            for k, v in catalog.items()
        ]
        return jsonify({
            'models': models_list,
            'recommendations': recommendations,
            'ai_note': ai_note,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Step 5: Train model ───────────────────────────────────────
@main.route('/api/pma/train', methods=['POST'])
def pma_train():
    data = request.get_json()
    file_path = session.get('pma_file') or session.get('uploaded_file') or data.get('file_path')
    target_col = data.get('target_col') or session.get('pma_target')
    model_key = data.get('model_key', 'random_forest')
    test_size = float(data.get('test_size', 0.2))
    if not file_path or not target_col:
        return jsonify({'error': 'Missing file or target column'}), 400
    try:
        df = load_file(file_path)
        df.columns = [str(c).strip() for c in df.columns]
        info = pma.detect_data_type(df, target_col)

        # Time series: add lag features
        if info['data_type'] == 'time_series' and info['date_cols']:
            df = pma.prepare_time_series(df, info['date_cols'][0], target_col)

        X, y, feature_names, encoders = pma.preprocess_tabular(df, target_col, info)

        # Split
        X_train, X_test, y_train, y_test = __import__('sklearn.model_selection', fromlist=['train_test_split']).train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        catalog = pma.get_model_catalog(info['data_type'], info['problem_type'])
        if model_key not in catalog:
            return jsonify({'error': f'Unknown model key: {model_key}'}), 400

        result = pma.train_selected_model(
            X_train, y_train, X_test, y_test,
            model_key, catalog, info['problem_type']
        )
        if 'error' in result:
            return jsonify({'error': result['error']}), 500

        # AI explanation of metrics
        ai_note = ''
        try:
            ai_note = _ml_agent.explain_metrics(
                result['metrics'], info['problem_type'], result['model_name']
            )
        except Exception:
            ai_note = 'AI explanation unavailable.'

        # Save model
        metadata = {
            'model_key': model_key,
            'model_name': result['model_name'],
            'target_col': target_col,
            'feature_names': feature_names,
            'problem_type': info['problem_type'],
            'data_type': info['data_type'],
            'metrics': result['metrics'],
        }
        model_path = pma.save_model_artifacts(result['model'], metadata, model_key)

        # Save encoders
        enc_path = model_path.replace('.pkl', '_encoders.pkl')
        with open(enc_path, 'wb') as f:
            pickle.dump(encoders, f)

        # Store in session
        session['pma_model_path'] = model_path
        session['pma_enc_path'] = enc_path
        session['pma_feature_names'] = feature_names
        session['pma_problem_type'] = info['problem_type']
        session['pma_target'] = target_col

        return jsonify({
            'metrics': result['metrics'],
            'logs': result.get('logs', []),
            'model_name': result['model_name'],
            'model_path': model_path,
            'feature_names': feature_names,
            'problem_type': info['problem_type'],
            'y_val': result.get('y_val', [])[:100],
            'y_pred': result.get('y_pred', [])[:100],
            'ai_note': ai_note,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Step 6: Hyperparameter tuning ────────────────────────────
@main.route('/api/pma/tune', methods=['POST'])
def pma_tune():
    data = request.get_json()
    file_path = session.get('pma_file') or session.get('uploaded_file') or data.get('file_path')
    target_col = data.get('target_col') or session.get('pma_target')
    model_key = data.get('model_key', 'random_forest')
    method = data.get('method', 'random')  # 'grid' or 'random'
    if not file_path or not target_col:
        return jsonify({'error': 'Missing file or target column'}), 400
    try:
        df = load_file(file_path)
        df.columns = [str(c).strip() for c in df.columns]
        info = pma.detect_data_type(df, target_col)
        X, y, feature_names, encoders = pma.preprocess_tabular(df, target_col, info)

        catalog = pma.get_model_catalog(info['data_type'], info['problem_type'])
        if model_key not in catalog or catalog[model_key]['model'] is None:
            return jsonify({'error': 'Cannot tune this model type'}), 400

        base_model = catalog[model_key]['model']
        tune_result = pma.tune_model(base_model, model_key, X, y, method, info['problem_type'])

        if 'error' in tune_result:
            return jsonify({'error': tune_result['error']}), 500

        # Retrain with best params if we got a best_model
        best_model = tune_result.get('best_model')
        if best_model:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            retrain = pma.train_selected_model(
                X_train, y_train, X_test, y_test,
                model_key,
                {model_key: {'name': catalog[model_key]['name'], 'model': best_model}},
                info['problem_type']
            )
            if 'metrics' in retrain:
                tune_result['tuned_metrics'] = retrain['metrics']
                # Save tuned model
                metadata = {
                    'model_key': model_key + '_tuned',
                    'model_name': catalog[model_key]['name'] + ' (Tuned)',
                    'target_col': target_col,
                    'feature_names': feature_names,
                    'problem_type': info['problem_type'],
                    'data_type': info['data_type'],
                    'metrics': retrain['metrics'],
                    'best_params': tune_result['best_params'],
                }
                model_path = pma.save_model_artifacts(best_model, metadata, model_key + '_tuned')
                enc_path = model_path.replace('.pkl', '_encoders.pkl')
                with open(enc_path, 'wb') as f:
                    pickle.dump(encoders, f)
                session['pma_model_path'] = model_path
                session['pma_enc_path'] = enc_path
                tune_result['model_path'] = model_path

        return jsonify({
            'best_params': tune_result.get('best_params', {}),
            'best_score': tune_result.get('best_score'),
            'tuned_metrics': tune_result.get('tuned_metrics', {}),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Step 7: Predict on new input ─────────────────────────────
@main.route('/api/pma/predict', methods=['POST'])
def pma_predict():
    data = request.get_json()
    input_data = data.get('input_data', {})
    model_path = data.get('model_path') or session.get('pma_model_path')
    enc_path = data.get('enc_path') or session.get('pma_enc_path')
    feature_names = data.get('feature_names') or session.get('pma_feature_names', [])
    problem_type = data.get('problem_type') or session.get('pma_problem_type', 'regression')
    if not model_path:
        return jsonify({'error': 'No model trained yet. Please train a model first.'}), 400
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        encoders = {}
        if enc_path and os.path.exists(enc_path):
            with open(enc_path, 'rb') as f:
                encoders = pickle.load(f)
        result = pma.predict_new_data(input_data, feature_names, model, encoders, problem_type)
        # AI insight
        ai_note = ''
        try:
            context = f'Prediction result: {result}, Input: {input_data}, Problem type: {problem_type}'
            ai_note = _ml_agent.run('Explain this prediction result to a non-technical user. What does it mean?', context)
        except Exception:
            pass
        result['ai_note'] = ai_note
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Step 8: List saved models ────────────────────────────────
@main.route('/api/pma/saved-models', methods=['GET'])
def pma_saved_models():
    meta_path = 'models/model_metadata.json'
    if not os.path.exists(meta_path):
        return jsonify({'models': []})
    try:
        with open(meta_path) as f:
            records = json.load(f)
        if not isinstance(records, list):
            records = []
        # Strip model object (not serializable), return metadata only
        safe = []
        for r in records:
            safe.append({
                'model_key': r.get('model_key'),
                'model_name': r.get('model_name'),
                'target_col': r.get('target_col'),
                'problem_type': r.get('problem_type'),
                'data_type': r.get('data_type'),
                'metrics': r.get('metrics', {}),
                'saved_at': r.get('saved_at'),
                'model_path': r.get('model_path'),
            })
        return jsonify({'models': safe})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Step 9: Load saved model into session ────────────────────
@main.route('/api/pma/load-model', methods=['POST'])
def pma_load_model():
    data = request.get_json()
    model_path = data.get('model_path')
    if not model_path or not os.path.exists(model_path):
        return jsonify({'error': 'Model file not found'}), 404
    try:
        meta_path = 'models/model_metadata.json'
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                records = json.load(f)
            for r in (records if isinstance(records, list) else []):
                if r.get('model_path') == model_path:
                    metadata = r
                    break
        enc_path = model_path.replace('.pkl', '_encoders.pkl')
        session['pma_model_path'] = model_path
        session['pma_enc_path'] = enc_path if os.path.exists(enc_path) else None
        session['pma_feature_names'] = metadata.get('feature_names', [])
        session['pma_problem_type'] = metadata.get('problem_type', 'regression')
        session['pma_target'] = metadata.get('target_col')
        return jsonify({'message': 'Model loaded into session', 'metadata': metadata})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Step 10: AI improvement suggestions ──────────────────────
@main.route('/api/pma/suggestions', methods=['POST'])
def pma_suggestions():
    data = request.get_json()
    metrics = data.get('metrics', {})
    model_key = data.get('model_key', '')
    try:
        info = session.get('pma_info', {})
        suggestion = _ml_agent.suggest_improvements(metrics, model_key, info)
        return jsonify({'suggestion': suggestion})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Step 11: Export predictions ───────────────────────────────
@main.route('/api/pma/export', methods=['POST'])
def pma_export():
    data = request.get_json()
    y_val = data.get('y_val', [])
    y_pred = data.get('y_pred', [])
    feature_names = data.get('feature_names', [])
    model_name = data.get('model_name', 'model')
    try:
        import io
        from flask import send_file
        rows = [{'actual': a, 'predicted': p} for a, p in zip(y_val, y_pred)]
        out_df = pd.DataFrame(rows)
        path = f'uploads/predictions_{model_name.replace(" ","_")}.csv'
        out_df.to_csv(path, index=False)
        return jsonify({'download_url': f'/{path}', 'rows': len(out_df)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ════════════════════════════════════════════════════════════
#  BIA — BUSINESS INTELLIGENCE & ANALYTICS ROUTES
# ════════════════════════════════════════════════════════════

# Helper to load the BIA working dataframe from session
def _bia_df():
    fp = session.get('bia_file') or session.get('uploaded_file')
    if not fp or not os.path.exists(fp):
        return None, 'No data file in session'
    try:
        df = load_file(fp)
        df.columns = [str(c).strip() for c in df.columns]
        return df, None
    except Exception as e:
        return None, str(e)


# ── Step 1: ETL — Extract ──────────────────────────────────────
@main.route('/api/bia/extract', methods=['POST'])
def bia_extract():
    data = request.get_json() or {}
    file_path = session.get('uploaded_file') or data.get('file_path')
    if not file_path:
        return jsonify({'error': 'No file in session. Please upload data first.'}), 400
    try:
        result = bia.extract_file(file_path)
        if 'error' in result:
            return jsonify(result), 500
        session['bia_file'] = file_path
        bia.log_event('extract', {'file': os.path.basename(file_path), 'rows': result['total_rows']})
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Step 1b: MySQL connection test ─────────────────────────
@main.route('/api/bia/mysql-test', methods=['POST'])
def bia_mysql_test():
    data = request.get_json() or {}
    result = bia.test_mysql_connection(
        data.get('host','localhost'), data.get('port', 3306),
        data.get('user','root'), data.get('password',''),
        data.get('database','')
    )
    if result['ok']:
        session['bia_mysql_cfg'] = data
    return jsonify(result)


# ── Step 1c: MySQL tables list ──────────────────────────────
@main.route('/api/bia/mysql-tables', methods=['POST'])
def bia_mysql_tables():
    cfg = session.get('bia_mysql_cfg') or request.get_json() or {}
    try:
        tables = bia.list_mysql_tables(cfg)
        return jsonify({'tables': tables})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Step 2: Transform ──────────────────────────────────────
@main.route('/api/bia/transform', methods=['POST'])
def bia_transform():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    data = request.get_json() or {}
    opts = {
        'drop_duplicates': data.get('drop_duplicates', True),
        'fill_missing': data.get('fill_missing', 'auto'),
        'normalize': data.get('normalize', False),
    }
    try:
        result = bia.transform_data(df, opts)
        clean_df = result['df']
        # Save transformed version
        fp = session.get('bia_file')
        if fp and fp.endswith('.csv'):
            clean_df.to_csv(fp, index=False)
        bia.log_event('transform', {'rows': len(clean_df), 'log': result['log']})
        return jsonify({
            'ok': True,
            'rows': len(clean_df),
            'columns': len(clean_df.columns),
            'log': result['log'],
            'num_cols': result['num_cols'],
            'cat_cols': result['cat_cols'],
            'date_cols': result['date_cols'],
            'preview': clean_df.head(8).to_dict(orient='records'),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Step 2b: MySQL load ─────────────────────────────────────
@main.route('/api/bia/mysql-load', methods=['POST'])
def bia_mysql_load():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    data = request.get_json() or {}
    cfg = session.get('bia_mysql_cfg') or data.get('cfg', {})
    table = data.get('table_name', 'bia_data')
    result = bia.load_to_mysql(df, cfg, table)
    if result['ok']:
        bia.log_event('mysql_load', {'table': table, 'rows': result['rows']})
    return jsonify(result)


# ── Step 3: KPIs ───────────────────────────────────────────
@main.route('/api/bia/kpis', methods=['POST'])
def bia_kpis():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    try:
        result = bia.compute_kpis(df)
        # Also compute aggregations if date+value found
        aggs = {}
        if result.get('date_col') and result.get('value_col'):
            aggs = bia.aggregate_data(df, result['date_col'], result['value_col'])
        result['aggregations'] = aggs
        session['bia_kpi_data'] = {k: v for k, v in result.items() if k != 'df'}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Step 4: EDA ────────────────────────────────────────────
@main.route('/api/bia/eda', methods=['POST'])
def bia_eda():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    try:
        result = bia.compute_eda(df)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Step 5: Chart data ──────────────────────────────────────
@main.route('/api/bia/chart', methods=['POST'])
def bia_chart():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    data = request.get_json() or {}
    try:
        result = bia.build_chart_data(
            df,
            chart_type=data.get('chart_type', 'bar'),
            x_col=data.get('x_col', ''),
            y_col=data.get('y_col', ''),
            color_col=data.get('color_col'),
            agg_func=data.get('agg_func', 'sum'),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Step 6: Dashboard data (all-in-one) ─────────────────────
@main.route('/api/bia/dashboard', methods=['POST'])
def bia_dashboard():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    try:
        kpi_data = bia.compute_kpis(df)
        eda_data = bia.compute_eda(df)
        insights = bia.generate_auto_insights(df, kpi_data)

        # Build charts
        charts = {}
        value_col = kpi_data.get('value_col')
        date_col  = kpi_data.get('date_col')
        cat_col   = kpi_data.get('cat_col')
        num_cols  = eda_data.get('num_cols', [])

        if date_col and value_col:
            agg = bia.aggregate_data(df, date_col, value_col)
            if 'monthly' in agg:
                charts['time_series'] = agg['monthly']

        if cat_col and value_col:
            charts['bar'] = bia.build_chart_data(df, 'bar', cat_col, value_col)
            charts['pie'] = bia.build_chart_data(df, 'pie', cat_col, value_col)

        if len(num_cols) >= 2:
            charts['heatmap'] = bia.build_chart_data(df, 'heatmap', num_cols[0], num_cols[1])

        if value_col:
            charts['histogram'] = bia.build_chart_data(df, 'histogram', value_col, value_col)

        return jsonify({
            'kpis': kpi_data['kpis'],
            'insights': insights,
            'charts': charts,
            'eda_summary': {
                'completeness': eda_data['completeness'],
                'n_rows': eda_data['n_rows'],
                'n_cols': eda_data['n_cols'],
            },
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Step 7: Advanced analytics ─────────────────────────────
@main.route('/api/bia/segment', methods=['POST'])
def bia_segment():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    data = request.get_json() or {}
    n_clusters = int(data.get('n_clusters', 4))
    result = bia.customer_segmentation(df, n_clusters)
    if result.get('ok'):
        bia.log_event('segmentation', {'clusters': n_clusters})
    return jsonify(result)


@main.route('/api/bia/anomalies', methods=['POST'])
def bia_anomalies():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    data = request.get_json() or {}
    value_col = data.get('value_col') or session.get('bia_kpi_data', {}).get('value_col')
    if not value_col:
        num_cols = df.select_dtypes(include='number').columns.tolist()
        value_col = num_cols[0] if num_cols else None
    if not value_col:
        return jsonify({'error': 'No numeric column found'}), 400
    result = bia.detect_anomalies(df, value_col)
    return jsonify(result)


@main.route('/api/bia/forecast', methods=['POST'])
def bia_forecast():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    data = request.get_json() or {}
    kpi_data = session.get('bia_kpi_data', {})
    date_col  = data.get('date_col')  or kpi_data.get('date_col')
    value_col = data.get('value_col') or kpi_data.get('value_col')
    periods   = int(data.get('periods', 12))
    if not date_col or not value_col:
        return jsonify({'error': 'Need date_col and value_col for forecasting'}), 400
    result = bia.time_series_forecast(df, date_col, value_col, periods)
    return jsonify(result)


# ── Step 8: AI Insights ─────────────────────────────────────
@main.route('/api/bia/insights', methods=['POST'])
def bia_insights():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    try:
        kpi_data = bia.compute_kpis(df)
        auto_insights = bia.generate_auto_insights(df, kpi_data)
        # LLM-powered insight (optional)
        ai_insight = ''
        try:
            from agents.ml_agent import MLAgent
            agent = MLAgent()
            context = f"""
            Dataset: {len(df)} rows x {len(df.columns)} cols
            KPIs: {json.dumps({k: v.get('value') for k,v in kpi_data['kpis'].items()})}
            Top insights detected: {[i['title'] for i in auto_insights]}
            """
            ai_insight = agent.run(
                'You are a business intelligence analyst. Based on the data context, provide 3 specific, actionable business insights in plain English. Focus on what actions the business should take.',
                context
            )
        except Exception:
            ai_insight = ''
        return jsonify({'insights': auto_insights, 'ai_insight': ai_insight})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Step 9: Natural Language Query (Ask your data) ─────────
@main.route('/api/bia/ask', methods=['POST'])
def bia_ask():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    data = request.get_json() or {}
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    try:
        num_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        context = f"""
Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns
Columns: {df.columns.tolist()}
Numeric columns: {num_cols}
Categorical columns: {cat_cols}
Sample data (first 3 rows):
{df.head(3).to_string()}
Basic stats:
{df.describe().to_string()}
"""
        response = orchestrator.run(question, session.get('bia_file'))
        bia.log_event('ask', {'question': question[:100]})
        return jsonify({'answer': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Step 10: Real-time refresh ──────────────────────────────
@main.route('/api/bia/refresh', methods=['POST'])
def bia_refresh():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    try:
        kpi_data = bia.compute_kpis(df)
        return jsonify({
            'kpis': kpi_data['kpis'],
            'rows': len(df),
            'refreshed_at': __import__('datetime').datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Step 11: Export ─────────────────────────────────────────
@main.route('/api/bia/export-csv', methods=['POST'])
def bia_export_csv():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    try:
        path = bia.export_to_csv(df, 'bia_export')
        return jsonify({'ok': True, 'download_url': f'/{path}', 'rows': len(df)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/bia/export-report', methods=['POST'])
def bia_export_report():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    try:
        kpi_data = bia.compute_kpis(df)
        insights = bia.generate_auto_insights(df, kpi_data)
        result = bia.build_pdf_report(kpi_data['kpis'], insights)
        if result['ok']:
            return jsonify({'ok': True, 'download_url': f'/{result["path"]}', 'note': result.get('note', '')})
        return jsonify({'error': result.get('error', 'Export failed')}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Step 12: Monitoring ──────────────────────────────────────
@main.route('/api/bia/monitor', methods=['GET'])
def bia_monitor():
    logs = bia.get_monitor_logs(50)
    health = bia.get_system_health()
    return jsonify({'logs': logs, 'health': health})


# ── BI page columns helper ──────────────────────────────────
@main.route('/api/bia/columns', methods=['POST'])
def bia_columns():
    df, err = _bia_df()
    if err: return jsonify({'error': err}), 400
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    return jsonify({
        'all_cols': df.columns.tolist(),
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'n_rows': len(df),
    })


# ════════════════════════════════════════════════════════════
#  AUTO DASHBOARD — Universal Data → Dashboard Generator
# ════════════════════════════════════════════════════════════
@main.route('/api/auto-dashboard', methods=['POST'])
def api_auto_dashboard():
    """Universal auto-dashboard: reads any uploaded file and returns
    all chart data, KPIs, stats, and insights in one shot."""
    data = request.get_json() or {}
    file_path = session.get('uploaded_file') or data.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'No data uploaded yet. Please upload a file first.'}), 400
    try:
        df = load_file(file_path)
        df.columns = [str(c).strip() for c in df.columns]

        num_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        all_cols = df.columns.tolist()
        n_rows, n_cols = df.shape

        # ── 1. KPIs ──────────────────────────────────────────────
        from src.bia import bia_engine as bia_e
        kpi_result = bia_e.compute_kpis(df)
        kpis = kpi_result.get('kpis', {})
        value_col = kpi_result.get('value_col')
        qty_col   = kpi_result.get('qty_col')
        date_col  = kpi_result.get('date_col')
        cat_col   = kpi_result.get('cat_col')

        # Always add fundamental stats KPIs
        kpis['total_rows']    = {'label': 'Total Records',  'value': n_rows,   'fmt': 'number', 'icon': '🗂️'}
        kpis['total_cols']    = {'label': 'Total Columns',  'value': n_cols,   'fmt': 'number', 'icon': '📋'}
        missing_pct = round(df.isnull().sum().sum() / (n_rows * n_cols) * 100, 1) if n_rows * n_cols else 0
        kpis['completeness']  = {'label': 'Completeness',   'value': round(100 - missing_pct, 1), 'fmt': 'percent', 'icon': '✅'}
        kpis['duplicates']    = {'label': 'Duplicate Rows', 'value': int(df.duplicated().sum()), 'fmt': 'number', 'icon': '⚠️'}

        # ── 2. Column profiles ───────────────────────────────────────
        col_profiles = []
        for col in all_cols:
            miss = int(df[col].isnull().sum())
            uniq = int(df[col].nunique())
            is_num = col in num_cols
            p = {'name': col, 'dtype': 'number' if is_num else 'text',
                 'missing': miss, 'missing_pct': round(miss/n_rows*100,1),
                 'unique': uniq}
            if is_num:
                s = df[col].dropna()
                if len(s):
                    p.update({
                        'min':    round(float(s.min()), 4),
                        'max':    round(float(s.max()), 4),
                        'mean':   round(float(s.mean()), 4),
                        'median': round(float(s.median()), 4),
                        'std':    round(float(s.std()), 4),
                    })
            else:
                vc = df[col].value_counts()
                p['top_val'] = str(vc.index[0]) if len(vc) else ''
                p['top_count'] = int(vc.iloc[0]) if len(vc) else 0
            col_profiles.append(p)

        # ── 3. Charts ───────────────────────────────────────────────
        charts = {}

        # Time-series line chart
        if date_col and value_col:
            try:
                df2 = df.copy()
                df2[date_col] = pd.to_datetime(df2[date_col], errors='coerce')
                df2 = df2.dropna(subset=[date_col])
                span = (df2[date_col].max() - df2[date_col].min()).days
                freq = 'YE' if span > 730 else ('ME' if span > 60 else 'W')
                agg  = df2.set_index(date_col)[value_col].resample(freq).sum().reset_index()
                fmt  = '%Y' if freq == 'YE' else ('%Y-%m' if freq == 'ME' else '%Y-%m-%d')
                charts['time_series'] = {
                    'labels': agg[date_col].dt.strftime(fmt).tolist(),
                    'values': [round(float(v), 2) for v in agg[value_col].tolist()],
                    'x_label': date_col, 'y_label': value_col,
                }
            except Exception:
                pass

        # Bar: top category by value
        if cat_col and value_col:
            top = df.groupby(cat_col)[value_col].sum().sort_values(ascending=False).head(15).reset_index()
            charts['bar'] = {
                'labels': top[cat_col].astype(str).tolist(),
                'values': top[value_col].round(2).tolist(),
                'x_label': cat_col, 'y_label': value_col,
            }

        # Donut / Pie
        if cat_col and value_col:
            pie = df.groupby(cat_col)[value_col].sum().sort_values(ascending=False).head(8).reset_index()
            charts['pie'] = {
                'labels': pie[cat_col].astype(str).tolist(),
                'values': pie[value_col].round(2).tolist(),
            }

        # Histogram for primary value col
        if value_col:
            s = df[value_col].dropna()
            counts, edges = np.histogram(s, bins=20)
            charts['histogram'] = {
                'labels': [round((edges[i]+edges[i+1])/2, 2) for i in range(len(counts))],
                'values': counts.tolist(), 'label': value_col,
            }

        # Scatter: top 2 numeric cols
        if len(num_cols) >= 2:
            c1, c2 = num_cols[0], num_cols[1]
            sample = df[[c1, c2]].dropna().sample(min(500, len(df)), random_state=42)
            charts['scatter'] = {
                'x': sample[c1].round(4).tolist(),
                'y': sample[c2].round(4).tolist(),
                'x_label': c1, 'y_label': c2,
            }

        # Correlation heatmap (up to 8 numeric cols)
        if len(num_cols) >= 2:
            hc = num_cols[:8]
            corr = df[hc].corr().round(3)
            charts['heatmap'] = {
                'cols': hc,
                'matrix': [[v if not np.isnan(v) else 0 for v in row] for row in corr.values.tolist()],
            }

        # Per-numeric-column: bar of stats
        numeric_summaries = []
        for col in num_cols[:10]:
            s = df[col].dropna()
            if not len(s): continue
            q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
            iqr = q3 - q1
            outliers = int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())
            numeric_summaries.append({
                'col': col, 'mean': round(float(s.mean()),3),
                'median': round(float(s.median()),3), 'std': round(float(s.std()),3),
                'min': round(float(s.min()),3), 'max': round(float(s.max()),3),
                'outliers': outliers,
            })

        # Per-cat-column: top value counts
        cat_summaries = []
        for col in cat_cols[:8]:
            vc = df[col].value_counts().head(8)
            cat_summaries.append({
                'col': col,
                'labels': vc.index.astype(str).tolist(),
                'counts': vc.values.tolist(),
            })

        # Extra: second category breakdowns
        extra_bars = []
        for cat in cat_cols[:4]:
            if cat == cat_col: continue
            if value_col:
                grp = df.groupby(cat)[value_col].sum().sort_values(ascending=False).head(10)
                extra_bars.append({
                    'col': cat, 'value_col': value_col,
                    'labels': grp.index.astype(str).tolist(),
                    'values': grp.round(2).tolist(),
                })

        # ── 4. Insights ──────────────────────────────────────────────
        insights = bia_e.generate_auto_insights(df, kpi_result)

        # ── 5. Data preview ──────────────────────────────────────────
        df_safe = df.where(pd.notnull(df), None)
        preview_rows = df_safe.head(10).to_dict(orient='records')
        for row in preview_rows:
            for k, v in row.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    row[k] = None

        return jsonify({
            'ok': True,
            'file_name': os.path.basename(file_path),
            'n_rows': n_rows,
            'n_cols': n_cols,
            'num_cols': num_cols,
            'cat_cols': cat_cols,
            'all_cols': all_cols,
            'kpis': kpis,
            'col_profiles': col_profiles,
            'charts': charts,
            'numeric_summaries': numeric_summaries,
            'cat_summaries': cat_summaries,
            'extra_bars': extra_bars,
            'insights': insights,
            'preview_rows': preview_rows,
            'detected': {
                'value_col': value_col, 'qty_col': qty_col,
                'date_col': date_col,   'cat_col': cat_col,
            },
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
