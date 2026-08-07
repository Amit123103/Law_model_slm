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
    <header className="h-16 px-4 border-b border-slate-800 bg-slate-900/95 backdrop-blur-md flex items-center justify-between z-20 shrink-0">
      <div className="flex items-center space-x-3">
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="p-2 rounded-xl hover:bg-slate-800 text-slate-300 md:hidden"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="flex items-center space-x-2.5">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center text-white shadow-md shadow-blue-500/20">
            <Scale className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="font-bold text-lg tracking-tight text-white">
                LawSLM
              </h1>
              <span className="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-blue-500/20 text-blue-400 border border-blue-500/30">
                v1.0 Scratch
              </span>
            </div>
            <p className="text-[11px] text-slate-400 hidden sm:block">
              Intelligent Legal & General Assistant
            </p>
          </div>
        </div>
      </div>

      <div className="flex items-center space-x-2 sm:space-x-3">
        {/* Active Model Status Badge */}
        <button 
          onClick={onOpenDashboard}
          className="hidden md:flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs font-medium text-slate-200 transition-colors"
        >
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span>{modelStats.checkpointLoaded.split('/').pop()}</span>
          <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
        </button>

        {/* System Prompt Button */}
        <button
          onClick={onOpenSystemPrompt}
          title="LawSLM System Prompt & Safety Directives"
          className="px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 transition-colors flex items-center space-x-1.5 text-xs font-medium"
        >
          <Sparkles className="w-4 h-4 text-amber-400" />
          <span className="hidden lg:inline">System Prompt</span>
        </button>

        {/* Dashboard Button */}
        <button
          onClick={onOpenDashboard}
          title="Model Health & Resource Monitor"
          className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-blue-400 transition-colors"
        >
          <Activity className="w-4 h-4" />
        </button>

        {/* Theme Toggle */}
        <button
          onClick={() => setDarkMode(!darkMode)}
          title="Toggle Light / Dark Theme"
          className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-amber-400 transition-colors"
        >
          {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4 text-slate-300" />}
        </button>

        {/* Settings Button */}
        <button
          onClick={onOpenSettings}
          title="Model Hyperparameters & Settings"
          className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 transition-colors"
        >
          <Settings className="w-4 h-4" />
        </button>

        {/* User Profile Avatar */}
        <div className="flex items-center space-x-2 pl-2 border-l border-slate-800">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white text-xs font-bold shadow-sm">
            AK
          </div>
          <div className="hidden xl:block text-left">
            <p className="text-xs font-semibold text-white">Amit Kumar</p>
            <p className="text-[10px] text-slate-400">Creator & Lead Eng</p>
          </div>
        </div>
      </div>
    </header>
  );
};
