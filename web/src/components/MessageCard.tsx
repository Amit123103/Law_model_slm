import React, { useState } from 'react';
import { 
  Scale, 
  User, 
  Copy, 
  Check, 
  ThumbsUp, 
  ThumbsDown, 
  RotateCw, 
  FileText, 
  Clock, 
  Cpu,
  Share2,
  Edit3,
  Trash2
} from 'lucide-react';
import type { Message } from '../types/chat';

interface MessageCardProps {
  message: Message;
  onRegenerate?: () => void;
  onOpenPDFPreview?: (pdf: NonNullable<Message['pdfPreview']>) => void;
  onEditMessage?: (id: string, text: string) => void;
  onDeleteMessage?: (id: string) => void;
}

export const MessageCard: React.FC<MessageCardProps> = ({
  message,
  onRegenerate,
  onOpenPDFPreview,
  onEditMessage,
  onDeleteMessage
}) => {
  const [copied, setCopied] = useState(false);
  const [liked, setLiked] = useState<boolean | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editText, setEditText] = useState(message.content);

  const isUser = message.role === 'user';

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleSaveEdit = () => {
    if (onEditMessage && editText.trim()) {
      onEditMessage(message.id, editText.trim());
      setIsEditing(false);
    }
  };

  // Simple Markdown & Table Parser
  const renderFormattedContent = (content: string) => {
    const codeBlockRegex = /```(\w*)\n([\s\S]*?)```/g;
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = codeBlockRegex.exec(content)) !== null) {
      if (match.index > lastIndex) {
        parts.push(content.substring(lastIndex, match.index));
      }
      const lang = match[1] || 'code';
      const code = match[2].trim();
      parts.push(
        <div key={match.index} className="my-4 rounded-2xl overflow-hidden border border-slate-700 bg-slate-950 text-slate-100 font-mono text-xs shadow-xl">
          <div className="flex items-center justify-between px-4 py-2 bg-slate-900 border-b border-slate-700 text-slate-400 font-semibold">
            <span>{lang}</span>
            <button
              onClick={() => {
                navigator.clipboard.writeText(code);
                setCopied(true);
                setTimeout(() => setCopied(false), 2000);
              }}
              className="flex items-center space-x-1.5 hover:text-white transition-colors"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>Copy Code</span>
            </button>
          </div>
          <pre className="p-4 overflow-x-auto">
            <code>{code}</code>
          </pre>
        </div>
      );
      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < content.length) {
      parts.push(content.substring(lastIndex));
    }

    return parts.map((p, idx) => {
      if (typeof p === 'string') {
        // Simple Markdown Table parsing
        if (p.includes('|')) {
          const lines = p.split('\n');
          const tableRows = lines.filter(l => l.trim().startsWith('|'));
          if (tableRows.length > 1) {
            return (
              <div key={idx} className="my-3 overflow-x-auto rounded-xl border border-slate-700 bg-slate-900/60 p-2">
                <table className="w-full text-xs text-left text-slate-200">
                  <tbody>
                    {tableRows.map((row, rIdx) => {
                      if (row.includes(':---') || row.includes('---')) return null;
                      const cols = row.split('|').filter(c => c.trim() !== '');
                      return (
                        <tr key={rIdx} className={rIdx === 0 ? 'bg-slate-800 font-bold text-white border-b border-slate-700' : 'border-b border-slate-800'}>
                          {cols.map((cell, cIdx) => (
                            <td key={cIdx} className="px-3 py-2">{cell.trim().replace(/\*\*/g, '')}</td>
                          ))}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            );
          }
        }
        return <p key={idx} className="whitespace-pre-wrap leading-relaxed">{p}</p>;
      }
      return p;
    });
  };

  return (
    <div className={`py-4 px-4 sm:px-6 w-full flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[85%] sm:max-w-[80%] rounded-2xl p-5 shadow-lg transition-all ${
        isUser 
          ? 'glass-card-user rounded-tr-none' 
          : 'glass-card-ai rounded-tl-none w-full'
      }`}>
        {/* Header Bar */}
        <div className="flex items-center justify-between mb-3 border-b border-slate-700/50 pb-2">
          <div className="flex items-center space-x-2.5">
            <div className={`w-7 h-7 rounded-xl flex items-center justify-center text-white shadow-sm ${
              isUser 
                ? 'bg-white/20' 
                : 'bg-gradient-to-br from-blue-600 to-indigo-600'
            }`}>
              {isUser ? <User className="w-4 h-4" /> : <Scale className="w-4 h-4" />}
            </div>
            <span className="text-xs font-bold text-white">
              {isUser ? 'You' : 'LawSLM Assistant'}
            </span>
          </div>

          <div className="flex items-center space-x-2 text-[11px] text-slate-400">
            <Clock className="w-3 h-3" />
            <span>{message.timestamp}</span>
          </div>
        </div>

        {/* Content Body */}
        {isEditing ? (
          <div className="space-y-2">
            <textarea
              value={editText}
              onChange={e => setEditText(e.target.value)}
              className="w-full p-3 text-sm rounded-xl bg-slate-900 border border-blue-500 text-white outline-none"
              rows={3}
            />
            <div className="flex justify-end space-x-2">
              <button
                onClick={() => setIsEditing(false)}
                className="px-3 py-1 rounded-lg bg-slate-700 text-xs text-slate-200"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveEdit}
                className="px-3 py-1 rounded-lg bg-blue-600 text-xs text-white font-semibold"
              >
                Save & Resend
              </button>
            </div>
          </div>
        ) : (
          <div className="text-sm text-slate-100 leading-relaxed font-normal">
            {renderFormattedContent(message.content)}
            {message.isStreaming && (
              <span className="streaming-cursor" />
            )}
          </div>
        )}

        {/* PDF Document Card if generated */}
        {message.pdfPreview && (
          <div className="mt-4 p-3.5 rounded-xl bg-blue-500/10 border border-blue-500/30 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <FileText className="w-6 h-6 text-blue-400" />
              <div>
                <h4 className="text-xs font-bold text-blue-300">{message.pdfPreview.title}</h4>
                <p className="text-[11px] text-slate-400">Formal Legal & Analytical PDF Report</p>
              </div>
            </div>
            <button
              onClick={() => onOpenPDFPreview && onOpenPDFPreview(message.pdfPreview!)}
              className="px-3.5 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold shadow-md transition-all"
            >
              Preview & Export PDF
            </button>
          </div>
        )}

        {/* Bottom Toolbar */}
        <div className="mt-3 pt-2.5 border-t border-slate-700/40 flex items-center justify-between text-xs text-slate-400">
          <div className="flex items-center space-x-2">
            <button
              onClick={handleCopy}
              className="flex items-center space-x-1 px-2.5 py-1 rounded-lg bg-slate-800/80 hover:bg-slate-700 border border-slate-700 text-slate-300 transition-colors"
              title="Copy text"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? 'Copied' : 'Copy'}</span>
            </button>

            {!isUser && (
              <>
                <button
                  onClick={() => setLiked(liked === true ? null : true)}
                  className={`p-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 border border-slate-700 transition-colors ${liked === true ? 'text-emerald-400 border-emerald-500/50' : 'text-slate-400'}`}
                  title="Good response"
                >
                  <ThumbsUp className="w-3.5 h-3.5" />
                </button>

                <button
                  onClick={() => setLiked(liked === false ? null : false)}
                  className={`p-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 border border-slate-700 transition-colors ${liked === false ? 'text-rose-400 border-rose-500/50' : 'text-slate-400'}`}
                  title="Bad response"
                >
                  <ThumbsDown className="w-3.5 h-3.5" />
                </button>

                {onRegenerate && (
                  <button
                    onClick={onRegenerate}
                    className="flex items-center space-x-1 px-2.5 py-1 rounded-lg bg-slate-800/80 hover:bg-slate-700 border border-slate-700 text-slate-300 transition-colors"
                    title="Regenerate answer"
                  >
                    <RotateCw className="w-3.5 h-3.5" />
                    <span>Retry</span>
                  </button>
                )}

                <button
                  onClick={handleCopy}
                  className="p-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 border border-slate-700 text-slate-400 transition-colors"
                  title="Share response"
                >
                  <Share2 className="w-3.5 h-3.5" />
                </button>
              </>
            )}

            {isUser && (
              <>
                <button
                  onClick={() => setIsEditing(true)}
                  className="p-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-white transition-colors"
                  title="Edit prompt"
                >
                  <Edit3 className="w-3.5 h-3.5" />
                </button>
                {onDeleteMessage && (
                  <button
                    onClick={() => onDeleteMessage(message.id)}
                    className="p-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-rose-300 transition-colors"
                    title="Delete message"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                )}
              </>
            )}
          </div>

          {!isUser && message.responseTimeMs && (
            <div className="flex items-center space-x-1.5 text-[10px] font-semibold text-slate-400">
              <Cpu className="w-3.5 h-3.5 text-blue-400" />
              <span>{(message.responseTimeMs / 1000).toFixed(2)}s</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
