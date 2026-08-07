import React, { useRef, useEffect } from 'react';
import { MessageCard } from './MessageCard';
import type { Message } from '../types/chat';
import { Scale, Sparkles, Code, FileText, Shield } from 'lucide-react';

interface ChatAreaProps {
  messages: Message[];
  onSendMessage: (text: string) => void;
  onRegenerate: () => void;
  onOpenPDFPreview: (pdf: NonNullable<Message['pdfPreview']>) => void;
}

export const ChatArea: React.FC<ChatAreaProps> = ({
  messages,
  onSendMessage,
  onRegenerate,
  onOpenPDFPreview
}) => {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 overflow-y-auto p-4 sm:p-8 flex flex-col items-center justify-center text-center">
        <div className="max-w-xl space-y-6">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center text-white mx-auto shadow-xl shadow-blue-500/25 animate-bounce">
            <Scale className="w-8 h-8" />
          </div>

          <div>
            <h2 className="text-2xl font-extrabold text-white tracking-tight">
              Welcome to LawSLM
            </h2>
            <p className="text-sm text-slate-400 mt-1">
              Your Intelligent AI Legal and General Assistant — Built completely from scratch by Amit Kumar.
            </p>
          </div>

          {/* Quick Start Suggestions */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-left">
            <SuggestionCard
              icon={<Shield className="w-5 h-5 text-blue-400" />}
              title="Legal Information"
              subtitle="Explain laws, procedure guidance & simple legal notices"
              onClick={() => onSendMessage("Explain Section 420 of IPC in simple terms")}
            />
            <SuggestionCard
              icon={<FileText className="w-5 h-5 text-amber-400" />}
              title="PDF Report Generation"
              subtitle="Create and export formal legal notices & affidavits"
              onClick={() => onSendMessage("Generate formal legal notice PDF report")}
            />
            <SuggestionCard
              icon={<Code className="w-5 h-5 text-emerald-400" />}
              title="Programming & AI"
              subtitle="Write, debug & explain Python, C++, Java & PyTorch"
              onClick={() => onSendMessage("Write a Python script to train a Transformer in PyTorch")}
            />
            <SuggestionCard
              icon={<Sparkles className="w-5 h-5 text-purple-400" />}
              title="Model Identity"
              subtitle="Learn about LawSLM's architecture and creator"
              onClick={() => onSendMessage("Who created you?")}
            />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto">
      {messages.map((msg, idx) => (
        <MessageCard
          key={msg.id || idx}
          message={msg}
          onRegenerate={idx === messages.length - 1 && msg.role === 'assistant' ? onRegenerate : undefined}
          onOpenPDFPreview={onOpenPDFPreview}
        />
      ))}
      <div ref={bottomRef} />
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
    className="p-4 rounded-2xl bg-slate-800/80 border border-slate-700 text-left flex flex-col justify-between space-y-2 hover:bg-slate-750 hover:border-blue-500/50 transition-all group"
  >
    <div className="p-2 rounded-xl bg-slate-900 w-fit group-hover:bg-blue-500/20 transition-colors">
      {icon}
    </div>
    <div>
      <h3 className="text-xs font-bold text-slate-200 group-hover:text-blue-400 transition-colors">
        {title}
      </h3>
      <p className="text-[11px] text-slate-400 mt-0.5 leading-snug">
        {subtitle}
      </p>
    </div>
  </button>
);
