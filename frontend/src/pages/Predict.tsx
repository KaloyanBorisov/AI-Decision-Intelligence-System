import React, { useEffect, useMemo, useState } from 'react';
import { getModels, predict, predictBatch, ModelSummary, PredictionResult } from '../services/modelService';
import { useToast } from '../context/ToastProvider';
import { Zap, Play, RotateCcw, TrendingUp } from 'lucide-react';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import SkeletonLoader from '../components/ui/SkeletonLoader';
import styles from './Predict.module.css';

interface SweepPoint {
    x: number;
    y: number;
}

// A prediction can be a raw regression value, a class label, or a
// classification with probabilities — reduce it to one plottable number:
// prefer the probability of the "positive"/last class (churn-, failure-,
// risk-style outputs), otherwise fall back to a numeric prediction.
const predictionToY = (result: PredictionResult): number | null => {
    if (result.probabilities && result.probabilities.length > 0) {
        return result.probabilities[result.probabilities.length - 1];
    }
    if (typeof result.prediction === 'number') return result.prediction;
    const num = Number(result.prediction);
    return Number.isFinite(num) ? num : null;
};

const Predict: React.FC = () => {
    const [models, setModels] = useState<ModelSummary[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedModelId, setSelectedModelId] = useState('');
    const [inputs, setInputs] = useState<Record<string, string>>({});
    const [running, setRunning] = useState(false);
    const [result, setResult] = useState<PredictionResult | null>(null);
    const { addToast } = useToast();

    // What-if sweep: vary one numeric feature across a range while holding
    // the rest of the form's inputs fixed, and chart how the prediction
    // responds — recreates the "curve" scenarios from the notebooks
    // (tool-wear degradation, price sensitivity, tenure decay, etc.).
    const [sweepFeature, setSweepFeature] = useState('');
    const [sweepMin, setSweepMin] = useState('0');
    const [sweepMax, setSweepMax] = useState('100');
    const [sweepSteps, setSweepSteps] = useState('12');
    const [sweepRunning, setSweepRunning] = useState(false);
    const [sweepPoints, setSweepPoints] = useState<SweepPoint[] | null>(null);

    useEffect(() => {
        const fetchModels = async () => {
            try {
                const data = await getModels();
                setModels(Array.isArray(data) ? data : []);
            } catch {
                addToast('Failed to load models', 'error');
            } finally {
                setLoading(false);
            }
        };
        fetchModels();
    }, []);

    const selectedModel = models.find(m => m.model_id === selectedModelId);
    const featureNames = selectedModel?.feature_names || [];

    const handleModelSelect = (modelId: string) => {
        setSelectedModelId(modelId);
        setResult(null);
        setSweepFeature('');
        setSweepPoints(null);
        const model = models.find(m => m.model_id === modelId);
        const initial: Record<string, string> = {};
        (model?.feature_names || []).forEach(f => { initial[f] = ''; });
        setInputs(initial);
    };

    const handleInputChange = (feature: string, value: string) => {
        setInputs(prev => ({ ...prev, [feature]: value }));
    };

    const handlePredict = async () => {
        if (!selectedModelId) return;
        setRunning(true);
        setResult(null);
        try {
            // Coerce numeric-looking inputs to numbers; leave the rest (e.g.
            // categorical text features) as strings for the model to handle.
            const data: Record<string, any> = {};
            for (const [key, val] of Object.entries(inputs)) {
                if (val.trim() === '') continue;
                const num = Number(val);
                data[key] = Number.isFinite(num) && val.trim() !== '' && !isNaN(num) ? num : val;
            }
            const prediction = await predict(selectedModelId, data);
            setResult(prediction);
        } catch (err: any) {
            const message = err?.response?.data?.detail || err?.message || 'Prediction failed';
            addToast(message, 'error');
        } finally {
            setRunning(false);
        }
    };

    const handleReset = () => {
        const cleared: Record<string, string> = {};
        featureNames.forEach(f => { cleared[f] = ''; });
        setInputs(cleared);
        setResult(null);
        setSweepPoints(null);
    };

    const handleSweep = async () => {
        if (!selectedModelId || !sweepFeature) return;
        const min = Number(sweepMin);
        const max = Number(sweepMax);
        const steps = Math.max(2, Math.min(50, Math.round(Number(sweepSteps)) || 0));
        if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) {
            addToast('Enter a valid min/max range (max must be greater than min)', 'error');
            return;
        }

        // Base record: whatever the user has already filled in, numeric
        // strings coerced to numbers, missing features defaulted to 0.
        const base: Record<string, any> = {};
        for (const f of featureNames) {
            const val = inputs[f];
            if (val === undefined || val.trim() === '') { base[f] = 0; continue; }
            const num = Number(val);
            base[f] = Number.isFinite(num) ? num : val;
        }

        const xs: number[] = [];
        for (let i = 0; i < steps; i++) {
            xs.push(min + ((max - min) * i) / (steps - 1));
        }
        const variants = xs.map(x => ({ ...base, [sweepFeature]: x }));

        setSweepRunning(true);
        setSweepPoints(null);
        try {
            const results = await predictBatch(selectedModelId, variants);
            const points: SweepPoint[] = xs
                .map((x, i) => ({ x, y: predictionToY(results[i]) }))
                .filter((p): p is SweepPoint => p.y !== null);
            if (points.length === 0) {
                addToast('Sweep ran, but predictions were not numeric/plottable', 'error');
            }
            setSweepPoints(points);
        } catch (err: any) {
            const message = err?.response?.data?.detail || err?.message || 'Sweep failed';
            addToast(message, 'error');
        } finally {
            setSweepRunning(false);
        }
    };

    const formatPrediction = (val: number | string): string => {
        if (typeof val === 'number') {
            return Number.isInteger(val) ? String(val) : val.toFixed(4);
        }
        return String(val);
    };

    // Chart geometry for the sweep line plot, in a fixed 600x220 viewBox.
    const sweepChart = useMemo(() => {
        if (!sweepPoints || sweepPoints.length === 0) return null;
        const width = 600;
        const height = 220;
        const pad = { top: 12, right: 12, bottom: 28, left: 44 };
        const xs = sweepPoints.map(p => p.x);
        const ys = sweepPoints.map(p => p.y);
        const xMin = Math.min(...xs), xMax = Math.max(...xs);
        const yMin = Math.min(0, ...ys), yMax = Math.max(...ys) || 1;
        const xScale = (x: number) =>
            pad.left + (xMax === xMin ? 0 : ((x - xMin) / (xMax - xMin)) * (width - pad.left - pad.right));
        const yScale = (y: number) =>
            height - pad.bottom - (yMax === yMin ? 0 : ((y - yMin) / (yMax - yMin)) * (height - pad.top - pad.bottom));
        const linePath = sweepPoints
            .map((p, i) => `${i === 0 ? 'M' : 'L'} ${xScale(p.x).toFixed(1)} ${yScale(p.y).toFixed(1)}`)
            .join(' ');
        return { width, height, pad, xMin, xMax, yMin, yMax, xScale, yScale, linePath };
    }, [sweepPoints]);

    return (
        <div className={styles.container}>
            <div className={styles.pageHeader}>
                <div>
                    <h1 className={styles.pageTitle}>Predict</h1>
                    <p className={styles.pageSubtitle}>Run inference against one of your trained models</p>
                </div>
            </div>

            {loading ? (
                <Card><SkeletonLoader variant="rect" height={300} /></Card>
            ) : models.length === 0 ? (
                <Card className={styles.emptyCard}>
                    <div className={styles.empty}>
                        <Zap size={40} className={styles.emptyIcon} />
                        <h3 className={styles.emptyTitle}>No trained models yet</h3>
                        <p className={styles.emptyDesc}>Train a model first, then come back here to run predictions</p>
                    </div>
                </Card>
            ) : (
                <div className={styles.layout}>
                    <Card className={styles.formCard}>
                        <div className={styles.formField}>
                            <label className={styles.formLabel}>Model</label>
                            <select
                                className={styles.formSelect}
                                value={selectedModelId}
                                onChange={(e) => handleModelSelect(e.target.value)}
                            >
                                <option value="">Select a model...</option>
                                {models.map((m) => (
                                    <option key={m.model_id} value={m.model_id}>
                                        {m.model_type} — {m.target_column}
                                    </option>
                                ))}
                            </select>
                        </div>

                        {selectedModel && (
                            <>
                                {featureNames.length === 0 ? (
                                    <p className={styles.noFeatures}>
                                        This model has no recorded feature list, so inputs can't be
                                        generated automatically.
                                    </p>
                                ) : (
                                    <div className={styles.featureGrid}>
                                        {featureNames.map((feature) => (
                                            <div key={feature} className={styles.formField}>
                                                <label className={styles.formLabel}>{feature}</label>
                                                <input
                                                    type="text"
                                                    className={styles.formInput}
                                                    placeholder="0"
                                                    value={inputs[feature] ?? ''}
                                                    onChange={(e) => handleInputChange(feature, e.target.value)}
                                                />
                                            </div>
                                        ))}
                                    </div>
                                )}

                                <div className={styles.formActions}>
                                    <Button
                                        variant="outline"
                                        size="sm"
                                        leftIcon={<RotateCcw size={14} />}
                                        onClick={handleReset}
                                        disabled={running}
                                    >
                                        Reset
                                    </Button>
                                    <Button
                                        variant="primary"
                                        size="sm"
                                        leftIcon={<Play size={14} />}
                                        onClick={handlePredict}
                                        loading={running}
                                        disabled={featureNames.length === 0}
                                    >
                                        {running ? 'Running...' : 'Run Prediction'}
                                    </Button>
                                </div>
                            </>
                        )}
                    </Card>

                    <Card className={styles.resultCard}>
                        <h3 className={styles.resultTitle}>Result</h3>
                        {!selectedModel ? (
                            <p className={styles.resultEmpty}>Select a model to get started</p>
                        ) : running ? (
                            <SkeletonLoader variant="rect" height={120} />
                        ) : !result ? (
                            <p className={styles.resultEmpty}>Fill in the inputs and run a prediction</p>
                        ) : (
                            <div className={styles.resultBody}>
                                <div className={styles.predictionValue}>
                                    {formatPrediction(result.prediction)}
                                </div>
                                <div className={styles.resultMeta}>
                                    <span>Model: {result.model}</span>
                                    {result.confidence !== null && (
                                        <span>Confidence: {Math.round(result.confidence * 100)}%</span>
                                    )}
                                </div>
                                {result.probabilities && result.probabilities.length > 0 && (
                                    <div className={styles.probList}>
                                        {result.probabilities.map((p, i) => (
                                            <div key={i} className={styles.probRow}>
                                                <span className={styles.probLabel}>Class {i}</span>
                                                <div className={styles.probBarTrack}>
                                                    <div
                                                        className={styles.probBarFill}
                                                        style={{ width: `${Math.round(p * 100)}%` }}
                                                    />
                                                </div>
                                                <span className={styles.probValue}>{Math.round(p * 100)}%</span>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}
                    </Card>
                </div>
            )}

            {selectedModel && featureNames.length > 0 && (
                <Card className={styles.sweepCard}>
                    <div className={styles.sweepHeader}>
                        <TrendingUp size={16} />
                        <div>
                            <h3 className={styles.resultTitle}>What-If Sweep</h3>
                            <p className={styles.resultEmpty}>
                                Vary one feature across a range and chart how the prediction responds —
                                the same pattern as the wear/price/tenure curves in the notebooks.
                            </p>
                        </div>
                    </div>

                    <div className={styles.sweepControls}>
                        <div className={styles.formField}>
                            <label className={styles.formLabel}>Feature to sweep</label>
                            <select
                                className={styles.formSelect}
                                value={sweepFeature}
                                onChange={(e) => setSweepFeature(e.target.value)}
                            >
                                <option value="">Select a feature...</option>
                                {featureNames.map((f) => (
                                    <option key={f} value={f}>{f}</option>
                                ))}
                            </select>
                        </div>
                        <div className={styles.formField}>
                            <label className={styles.formLabel}>Min</label>
                            <input
                                type="number"
                                className={styles.formInput}
                                value={sweepMin}
                                onChange={(e) => setSweepMin(e.target.value)}
                            />
                        </div>
                        <div className={styles.formField}>
                            <label className={styles.formLabel}>Max</label>
                            <input
                                type="number"
                                className={styles.formInput}
                                value={sweepMax}
                                onChange={(e) => setSweepMax(e.target.value)}
                            />
                        </div>
                        <div className={styles.formField}>
                            <label className={styles.formLabel}>Steps</label>
                            <input
                                type="number"
                                className={styles.formInput}
                                value={sweepSteps}
                                onChange={(e) => setSweepSteps(e.target.value)}
                                min={2}
                                max={50}
                            />
                        </div>
                        <Button
                            variant="primary"
                            size="sm"
                            leftIcon={<Play size={14} />}
                            onClick={handleSweep}
                            loading={sweepRunning}
                            disabled={!sweepFeature}
                        >
                            {sweepRunning ? 'Running...' : 'Run Sweep'}
                        </Button>
                    </div>

                    {sweepChart && (
                        <div className={styles.sweepChartWrap}>
                            <svg
                                viewBox={`0 0 ${sweepChart.width} ${sweepChart.height}`}
                                className={styles.sweepSvg}
                                role="img"
                                aria-label={`Predicted output vs ${sweepFeature}`}
                            >
                                {/* Y axis */}
                                <line
                                    x1={sweepChart.pad.left} y1={sweepChart.pad.top}
                                    x2={sweepChart.pad.left} y2={sweepChart.height - sweepChart.pad.bottom}
                                    className={styles.sweepAxis}
                                />
                                {/* X axis */}
                                <line
                                    x1={sweepChart.pad.left} y1={sweepChart.height - sweepChart.pad.bottom}
                                    x2={sweepChart.width - sweepChart.pad.right} y2={sweepChart.height - sweepChart.pad.bottom}
                                    className={styles.sweepAxis}
                                />
                                <path d={sweepChart.linePath} className={styles.sweepLine} />
                                {sweepPoints!.map((p, i) => (
                                    <circle
                                        key={i}
                                        cx={sweepChart.xScale(p.x)}
                                        cy={sweepChart.yScale(p.y)}
                                        r={3}
                                        className={styles.sweepDot}
                                    />
                                ))}
                                <text x={sweepChart.pad.left} y={sweepChart.height - 6} className={styles.sweepAxisLabel}>
                                    {sweepChart.xMin.toFixed(1)}
                                </text>
                                <text x={sweepChart.width - sweepChart.pad.right} y={sweepChart.height - 6} textAnchor="end" className={styles.sweepAxisLabel}>
                                    {sweepChart.xMax.toFixed(1)}
                                </text>
                                <text x={sweepChart.pad.left - 6} y={sweepChart.pad.top + 8} textAnchor="end" className={styles.sweepAxisLabel}>
                                    {sweepChart.yMax.toFixed(2)}
                                </text>
                                <text x={sweepChart.pad.left - 6} y={sweepChart.height - sweepChart.pad.bottom} textAnchor="end" className={styles.sweepAxisLabel}>
                                    {sweepChart.yMin.toFixed(2)}
                                </text>
                            </svg>
                            <p className={styles.sweepCaption}>
                                {sweepFeature} (x-axis) vs. predicted output (y-axis)
                            </p>
                        </div>
                    )}
                </Card>
            )}
        </div>
    );
};

export default Predict;
