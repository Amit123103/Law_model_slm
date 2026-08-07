import React from 'react';
import { X, Activity, Cpu, HardDrive, Database, Layers, CheckCircle, RefreshCw } from 'lucide-react';
import { ModelStats } from '../types/chat';

interface DashboardModalProps {
  isOpen: boolean;
  onClose: () => void;
  stats: ModelStats;
}

export const DashboardModal: React.FC<DashboardModalProps> = ({
  isOpen,
  onClose,
  stats
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-fade-in">
      <div className="w-full max-w-2xl glass-panel rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <Activity className="w-5 h-5 text-brand-500" />
            <div>
              <h2 className="text-base font-bold text-slate-900 dark:text-white">LawSLM System Dashboard</h2>
              <p className="text-[11px] text-slate-500 dark:text-slate-400">Real-time Architecture & Health Monitoring</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-xl text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Dashboard Grid Content */}
        <div className="p-6 overflow-y-auto space-y-6 flex-1 text-xs">
          {/* Status Metric Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <MetricBox icon={<Activity className="w-4 h-4 text-emerald-500" />} label="Status" value="Online & Active" />
            <MetricBox icon={<Cpu className="w-4 h-4 text-brand-500" />} label="Execution Device" value={stats.activeDevice} />
            <MetricBox icon={<Database className="w-4 h-4 text-indigo-500" />} label="Total Parameters" value={stats.totalParams} />
            <MetricBox icon={<HardDrive className="w-4 h-4 text-amber-500" />} label="RAM / VRAM" value={stats.ramUsage} />
          </div>

          {/* Architecture Details */}
          <div className="p-4 rounded-xl bg-slate-100/70 dark:bg-darkbg-900/70 border border-slate-200 dark:border-slate-800 space-y-3">
            <div className="flex items-center justify-between font-bold text-slate-800 dark:text-slate-200 border-b border-slate-200 dark:border-slate-800 pb-2">
              <span className="flex items-center space-x-1.5">
                <Layers className="w-4 h-4 text-brand-500" />
                <span>Transformer Architectural Hyperparameters</span>
              </span>
              <span className="px-2 py-0.5 rounded bg-brand-100 dark:bg-brand-500/20 text-brand-600 dark:text-brand-400 text-[10px]">
                Built 100% From Scratch
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-slate-600 dark:text-slate-300">
              <div><span className="font-semibold text-slate-900 dark:text-white">Vocab Size:</span> {stats.vocabSize}</div>
              <div><span className="font-semibold text-slate-900 dark:text-white">Embedding Dim (d):</span> {stats.dModel}</div>
              <div><span className="font-semibold text-slate-900 dark:text-white">Layers (N):</span> {stats.nLayers}</div>
              <div><span className="font-semibold text-slate-900 dark:text-white">Attention Heads:</span> {stats.nHeads}</div>
              <div><span className="font-semibold text-slate-900 dark:text-white">Normalization:</span> RMSNorm</div>
              <div><span className="font-semibold text-slate-900 dark:text-white">Pos Encoding:</span> RoPE</div>
            </div>
          </div>

          {/* Active Checkpoint Details */}
          <div className="p-4 rounded-xl bg-slate-100/70 dark:bg-darkbg-900/70 border border-slate-200 dark:border-slate-800 space-y-2">
            <h4 className="font-bold text-slate-800 dark:text-slate-200">Loaded Checkpoint State</h4>
            <div className="flex items-center justify-between text-slate-600 dark:text-slate-300 font-mono text-[11px]">
              <span>Path: {stats.checkpointLoaded}</span>
              <span className="flex items-center space-x-1 text-emerald-500 font-semibold">
                <CheckCircle className="w-3.5 h-3.5" />
                <span>Verified</span>
              </span>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-slate-200 dark:border-slate-800 flex justify-between items-center text-[11px] text-slate-400">
          <span>LawSLM MLOps Telemetry</span>
          <span>Lead Engineer: Amit Kumar</span>
        </div>
      </div>
    </div>
  );
};

const MetricBox: React.FC<{ icon: React.ReactNode; label: string; value: string }> = ({ icon, label, value }) => (
  <div className="p-3 rounded-xl bg-white dark:bg-darkbg-800 border border-slate-200 dark:border-slate-700 shadow-sm flex flex-col space-y-1">
    <div className="flex items-center space-x-1.5 text-slate-400">
      {icon}
      <span className="text-[10px] uppercase font-bold tracking-wider">{label}</span>
    </div>
    <span className="text-xs font-bold text-slate-900 dark:text-white">{value}</span>
  </div>
);
