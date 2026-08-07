import React from 'react';
import { X, Sliders, Sparkles } from 'lucide-react';
import type { ModelParams } from '../types/chat';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  params: ModelParams;
  onUpdateParams: (newParams: ModelParams) => void;
  systemPrompt: string;
  onUpdateSystemPrompt: (prompt: string) => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  params,
  onUpdateParams,
  systemPrompt,
  onUpdateSystemPrompt
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 backdrop-blur-sm animate-fade-in">
      <div className="w-full max-w-xl glass-panel rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <Sliders className="w-5 h-5 text-brand-500" />
            <h2 className="text-base font-bold text-slate-900 dark:text-white">Model Parameters & Hyperparameters</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-xl text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-6 flex-1 text-xs">
          {/* Temperature Slider */}
          <div className="space-y-2">
            <div className="flex items-center justify-between font-semibold text-slate-700 dark:text-slate-200">
              <span>Temperature (Creativity)</span>
              <span className="px-2 py-0.5 rounded bg-brand-50 dark:bg-brand-500/20 text-brand-600 dark:text-brand-400 font-mono">
                {params.temperature}
              </span>
            </div>
            <input
              type="range"
              min="0.0"
              max="2.0"
              step="0.05"
              value={params.temperature}
              onChange={e => onUpdateParams({ ...params, temperature: parseFloat(e.target.value) })}
              className="w-full accent-brand-500 cursor-pointer"
            />
            <p className="text-[11px] text-slate-400">
              0.0 = Deterministic Greedy Decoding (Exact & Precise). 0.8 = Creative Sampling.
            </p>
          </div>

          {/* Top-K Slider */}
          <div className="space-y-2">
            <div className="flex items-center justify-between font-semibold text-slate-700 dark:text-slate-200">
              <span>Top-K Limit</span>
              <span className="px-2 py-0.5 rounded bg-brand-50 dark:bg-brand-500/20 text-brand-600 dark:text-brand-400 font-mono">
                {params.topK}
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              step="1"
              value={params.topK}
              onChange={e => onUpdateParams({ ...params, topK: parseInt(e.target.value) })}
              className="w-full accent-brand-500 cursor-pointer"
            />
            <p className="text-[11px] text-slate-400">
              Filters sampling to the top K highest-probability tokens.
            </p>
          </div>

          {/* Top-P Slider */}
          <div className="space-y-2">
            <div className="flex items-center justify-between font-semibold text-slate-700 dark:text-slate-200">
              <span>Top-P (Nucleus Threshold)</span>
              <span className="px-2 py-0.5 rounded bg-brand-50 dark:bg-brand-500/20 text-brand-600 dark:text-brand-400 font-mono">
                {params.topP}
              </span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={params.topP}
              onChange={e => onUpdateParams({ ...params, topP: parseFloat(e.target.value) })}
              className="w-full accent-brand-500 cursor-pointer"
            />
          </div>

          {/* Max Tokens Slider */}
          <div className="space-y-2">
            <div className="flex items-center justify-between font-semibold text-slate-700 dark:text-slate-200">
              <span>Max New Generation Tokens</span>
              <span className="px-2 py-0.5 rounded bg-brand-50 dark:bg-brand-500/20 text-brand-600 dark:text-brand-400 font-mono">
                {params.maxTokens}
              </span>
            </div>
            <input
              type="range"
              min="16"
              max="512"
              step="16"
              value={params.maxTokens}
              onChange={e => onUpdateParams({ ...params, maxTokens: parseInt(e.target.value) })}
              className="w-full accent-brand-500 cursor-pointer"
            />
          </div>

          {/* System Prompt Customizer */}
          <div className="space-y-2 pt-2 border-t border-slate-200 dark:border-slate-800">
            <div className="flex items-center space-x-2 font-semibold text-slate-700 dark:text-slate-200">
              <Sparkles className="w-4 h-4 text-amber-500" />
              <span>Active System Prompt</span>
            </div>
            <textarea
              rows={4}
              value={systemPrompt}
              onChange={e => onUpdateSystemPrompt(e.target.value)}
              className="w-full p-3 rounded-xl bg-slate-100 dark:bg-darkbg-900 border border-slate-200 dark:border-slate-700 text-xs font-mono outline-none focus:border-brand-500"
            />
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3 border-t border-slate-200 dark:border-slate-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-brand-500 hover:bg-brand-600 text-white font-medium text-xs transition-colors shadow-md"
          >
            Save Parameters
          </button>
        </div>
      </div>
    </div>
  );
};
