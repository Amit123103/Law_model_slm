import React from 'react';
import { 
  Scale, 
  Sun, 
  Moon, 
  Settings, 
  Sparkles, 
  Activity,
  Menu,
  ChevronDown
} from 'lucide-react';
import type { ModelStats } from '../types/chat';

interface HeaderProps {
  darkMode: boolean;
  setDarkMode: (val: boolean) => void;
  onOpenSettings: () => void;
  onOpenDashboard: () => void;
  onOpenSystemPrompt: () => void;
  modelStats: ModelStats;
  sidebarOpen: boolean;
  setSidebarOpen: (val: boolean) => void;
}

export const Header: React.FC<HeaderProps> = ({
  darkMode,
  setDarkMode,
  onOpenSettings,
  onOpenDashboard,
  onOpenSystemPrompt,
  modelStats,
  sidebarOpen,
  setSidebarOpen
}) => {
  return (
    <header className="h-[72px] px-6 glass-header flex items-center justify-between z-30 shrink-0 sticky top-0 w-full">
      <div className="flex items-center space-x-4">
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="p-2 rounded-xl hover:bg-slate-800 text-slate-300 md:hidden transition-colors"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center text-white shadow-lg shadow-blue-500/25">
            <Scale className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="font-extrabold text-lg tracking-tight text-white">
                LawSLM
              </h1>
              <span className="px-2.5 py-0.5 text-[10px] font-bold rounded-full bg-blue-500/20 text-blue-400 border border-blue-500/30 tracking-wider">
                v1.0 Scratch
              </span>
            </div>
            <p className="text-[11px] text-slate-400 font-medium hidden sm:block">
              Intelligent Legal & General Assistant
            </p>
          </div>
        </div>
      </div>

      <div className="flex items-center space-x-2 sm:space-x-3">
        {/* Model Selector Dropdown Badge */}
        <button 
          onClick={onOpenDashboard}
          className="hidden md:flex items-center space-x-2 px-3.5 py-2 rounded-xl bg-slate-800/90 hover:bg-slate-750 border border-slate-700 text-xs font-medium text-slate-200 transition-all shadow-sm"
        >
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
          <span className="font-semibold">{modelStats.checkpointLoaded.split('/').pop()}</span>
          <ChevronDown className="w-3.5 h-3.5 text-slate-400 ml-1" />
        </button>

        {/* System Prompt Modal Trigger */}
        <button
          onClick={onOpenSystemPrompt}
          title="LawSLM System Prompt & Safety Directives"
          className="px-3.5 py-2 rounded-xl bg-slate-800/90 hover:bg-slate-750 border border-slate-700 text-slate-200 transition-all flex items-center space-x-2 text-xs font-medium shadow-sm"
        >
          <Sparkles className="w-4 h-4 text-amber-400" />
          <span className="hidden lg:inline font-semibold">System Prompt</span>
        </button>

        {/* Performance Monitor */}
        <button
          onClick={onOpenDashboard}
          title="Model Health & Resource Monitor"
          className="p-2.5 rounded-xl bg-slate-800/90 hover:bg-slate-750 border border-slate-700 text-blue-400 transition-all shadow-sm"
        >
          <Activity className="w-4 h-4" />
        </button>

        {/* Theme Toggle */}
        <button
          onClick={() => setDarkMode(!darkMode)}
          title="Toggle Light / Dark Theme"
          className="p-2.5 rounded-xl bg-slate-800/90 hover:bg-slate-750 border border-slate-700 text-amber-400 transition-all shadow-sm"
        >
          {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4 text-slate-300" />}
        </button>

        {/* Settings Modal */}
        <button
          onClick={onOpenSettings}
          title="Model Hyperparameters & Settings"
          className="p-2.5 rounded-xl bg-slate-800/90 hover:bg-slate-750 border border-slate-700 text-slate-300 transition-all shadow-sm"
        >
          <Settings className="w-4 h-4" />
        </button>

        {/* User Profile Avatar */}
        <div className="flex items-center space-x-2.5 pl-3 border-l border-slate-800">
          <div className="w-9 h-9 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white text-xs font-bold shadow-md">
            AK
          </div>
          <div className="hidden xl:block text-left">
            <p className="text-xs font-bold text-white leading-none">Amit Kumar</p>
            <p className="text-[10px] text-slate-400 mt-1 font-medium">Creator & Lead Eng</p>
          </div>
        </div>
      </div>
    </header>
  );
};
