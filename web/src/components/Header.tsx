import React from 'react';
import { 
  Scale, 
  Sun, 
  Moon, 
  Settings, 
  LayoutDashboard, 
  Sparkles, 
  FileText, 
  Activity,
  Menu,
  ChevronDown
} from 'lucide-react';
import { ModelStats } from '../types/chat';

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
    <header className="h-16 px-4 border-b border-slate-200 dark:border-slate-800 glass-panel flex items-center justify-between z-20">
      <div className="flex items-center space-x-3">
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="p-2 rounded-xl hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 md:hidden"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="flex items-center space-x-2.5">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-brand-500 to-indigo-600 flex items-center justify-center text-white shadow-md shadow-brand-500/20">
            <Scale className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="font-bold text-lg tracking-tight bg-gradient-to-r from-slate-900 via-brand-600 to-indigo-600 dark:from-white dark:via-brand-400 dark:to-indigo-400 bg-clip-text text-transparent">
                LawSLM
              </h1>
              <span className="px-2 py-0.5 text-xs font-semibold rounded-full bg-brand-100 dark:bg-brand-500/20 text-brand-600 dark:text-brand-400 border border-brand-200 dark:border-brand-500/30">
                v1.0 Scratch
              </span>
            </div>
            <p className="text-[11px] text-slate-500 dark:text-slate-400 hidden sm:block">
              Intelligent Legal & General Assistant
            </p>
          </div>
        </div>
      </div>

      <div className="flex items-center space-x-2 sm:space-x-3">
        {/* Active Model Status Badge */}
        <button 
          onClick={onOpenDashboard}
          className="hidden md:flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-xs font-medium text-slate-700 dark:text-slate-300 transition-colors"
        >
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span>{modelStats.checkpointLoaded.split('/').pop()}</span>
          <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
        </button>

        {/* System Prompt Button */}
        <button
          onClick={onOpenSystemPrompt}
          title="LawSLM System Prompt & Safety Directives"
          className="p-2 rounded-xl text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors flex items-center space-x-1 text-xs font-medium"
        >
          <Sparkles className="w-4 h-4 text-amber-500" />
          <span className="hidden lg:inline">System Prompt</span>
        </button>

        {/* Dashboard Button */}
        <button
          onClick={onOpenDashboard}
          title="Model Health & Resource Monitor"
          className="p-2 rounded-xl text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
        >
          <Activity className="w-4 h-4 text-brand-500" />
        </button>

        {/* Theme Toggle */}
        <button
          onClick={() => setDarkMode(!darkMode)}
          title="Toggle Light / Dark Theme"
          className="p-2 rounded-xl text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
        >
          {darkMode ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-slate-600" />}
        </button>

        {/* Settings Button */}
        <button
          onClick={onOpenSettings}
          title="Model Hyperparameters & Settings"
          className="p-2 rounded-xl text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
        >
          <Settings className="w-4 h-4" />
        </button>

        {/* User Profile Avatar */}
        <div className="flex items-center space-x-2 pl-2 border-l border-slate-200 dark:border-slate-800">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white text-xs font-bold shadow-sm">
            AK
          </div>
          <div className="hidden xl:block text-left">
            <p className="text-xs font-semibold text-slate-800 dark:text-slate-200">Amit Kumar</p>
            <p className="text-[10px] text-slate-500 dark:text-slate-400">Creator & Lead Eng</p>
          </div>
        </div>
      </div>
    </header>
  );
};
