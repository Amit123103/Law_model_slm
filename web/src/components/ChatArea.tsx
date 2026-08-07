import React, { useRef, useEffect } from 'react';
import { MessageCard } from './MessageCard';
import type { Message } from '../types/chat';
import { Scale, Sparkles, Code, FileText, Shield, Loader2 } from 'lucide-react';

interface ChatAreaProps {
  messages: Message[];
  onSendMessage: (text: string) => void;
  onRegenerate: () => void;
  onOpenPDFPreview: (pdf: NonNullable<Message['pdfPreview']>) => void;
  onEditMessage?: (id: string, text: string) => void;
  onDeleteMessage?: (id: string) => void;
  isGenerating?: boolean;
}

export const ChatArea: React.FC<ChatAreaProps> = ({
  messages,
  onSendMessage,
  onRegenerate,
  onOpenPDFPreview,
  onEditMessage,
  onDeleteMessage,
  isGenerating
}) => {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isGenerating]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 overflow-y-auto p-6 sm:p-12 flex flex-col items-center justify-center text-center">
        <div className="max-w-[900px] w-full mx-auto space-y-8">
          {/* Logo & Header */}
          <div className="space-y-3">
            <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center text-white mx-auto shadow-2xl shadow-blue-500/30 animate-pulse">
              <Scale className="w-10 h-10" />
            </div>

            <h2 className="text-3xl font-extrabold text-white tracking-tight">
              Welcome to LawSLM
            </h2>
            <p className="text-sm text-slate-400 max-w-lg mx-auto">
              Your Intelligent AI Legal and General Assistant — Built completely from scratch by Amit Kumar.
            </p>
          </div>

          {/* 4 Feature Suggestion Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-left max-w-2xl mx-auto">
            <SuggestionCard
              icon={<Shield className="w-6 h-6 text-blue-400" />}
              title="Legal Information"
              subtitle="Explain statutes, IPC section provisions & legal procedures"
              onClick={() => onSendMessage("Explain Section 420 of IPC in simple terms")}
            />
            <SuggestionCard
              icon={<FileText className="w-6 h-6 text-amber-400" />}
              title="PDF Report Export"
              subtitle="Generate & export formal legal notices & affidavit documents"
              onClick={() => onSendMessage("Generate formal legal notice PDF report")}
            />
            <SuggestionCard
              icon={<Code className="w-6 h-6 text-emerald-400" />}
              title="Programming & AI"
              subtitle="Write & debug Python, PyTorch, C++, Java & SQL code"
              onClick={() => onSendMessage("Write a Python script to train a Transformer in PyTorch")}
            />
            <SuggestionCard
              icon={<Sparkles className="w-6 h-6 text-purple-400" />}
              title="Model Creator"
              subtitle="Learn about LawSLM's architecture and developer"
              onClick={() => onSendMessage("Who created you?")}
            />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto py-6">
      <div className="max-w-[900px] mx-auto space-y-4">
        {messages.map((msg, idx) => (
          <MessageCard
            key={msg.id || idx}
            message={msg}
            onRegenerate={idx === messages.length - 1 && msg.role === 'assistant' ? onRegenerate : undefined}
            onOpenPDFPreview={onOpenPDFPreview}
            onEditMessage={onEditMessage}
            onDeleteMessage={onDeleteMessage}
          />
        ))}

        {/* Animated Thinking Indicator */}
        {isGenerating && messages[messages.length - 1]?.role === 'user' && (
          <div className="flex items-center space-x-3 px-6 py-4 rounded-2xl glass-card-ai max-w-xs mx-6">
            <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
            <span className="text-xs font-semibold text-slate-300">LawSLM is thinking...</span>
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  );
};

interface SuggestionCardProps {
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  onClick: () => void;
}

const SuggestionCard: React.FC<SuggestionCardProps> = ({ icon, title, subtitle, onClick }) => (
  <button
    onClick={onClick}
    className="p-5 rounded-2xl bg-slate-800/80 border border-slate-700/80 text-left flex flex-col justify-between space-y-3 hover:bg-slate-750 hover:border-blue-500/50 transition-all shadow-lg group"
  >
    <div className="p-2.5 rounded-xl bg-slate-900 w-fit group-hover:bg-blue-500/20 transition-colors">
      {icon}
    </div>
    <div>
      <h3 className="text-sm font-bold text-slate-200 group-hover:text-blue-400 transition-colors">
        {title}
      </h3>
      <p className="text-xs text-slate-400 mt-1 leading-snug">
        {subtitle}
      </p>
    </div>
  </button>
);
