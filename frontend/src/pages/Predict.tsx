import React, { useEffect, useState } from 'react';
import { getModels, predict, ModelSummary, PredictionResult } from '../services/modelService';
import { useToast } from '../context/ToastProvider';
import { Zap, Play, RotateCcw } from 'lucide-react';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import SkeletonLoader from '../components/ui/SkeletonLoader';
import styles from './Predict.module.css';

const Predict: React.FC = () => {
    const [models, setModels] = useState<ModelSummary[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedModelId, setSelectedModelId] = useState('');
    const [inputs, setInputs] = useState<Record<string, string>>({});
    const [running, setRunning] = useState(false);
    const [result, setResult] = useState<PredictionResult | null>(null);
    const { addToast } = useToast();

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
    };

    const formatPrediction = (val: number | string): string => {
        if (typeof val === 'number') {
            return Number.isInteger(val) ? String(val) : val.toFixed(4);
        }
        return String(val);
    };

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
        </div>
    );
};

export default Predict;
