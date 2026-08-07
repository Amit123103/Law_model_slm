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
  Sparkles, 
  Clock, 
  Cpu
} from 'lucide-react';
import { Message } from '../types/chat';

interface MessageCardProps {
  message: Message;
  onRegenerate?: () => void;
  onOpenPDFPreview?: (pdf: NonNullable<Message['pdfPreview']>) => void;
}

export const MessageCard: React.FC<MessageCardProps> = ({
  message,
  onRegenerate,
  onOpenPDFPreview
}) => {
  const [copied, setCopied] = useState(false);
  const [liked, setLiked] = useState<boolean | null>(null);

  const isUser = message.role === 'user';

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Simple Markdown parsing for code blocks and bold text
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
        <div key={match.index} className="my-3 rounded-xl overflow-hidden border border-slate-700 bg-slate-900 text-slate-100 font-mono text-xs shadow-lg">
          <div className="flex items-center justify-between px-3 py-1.5 bg-slate-800 border-b border-slate-700 text-slate-400">
            <span>{lang}</span>
            <button
              onClick={() => {
                navigator.clipboard.writeText(code);
                setCopied(true);
                setTimeout(() => setCopied(false), 2000);
              }}
              className="flex items-center space-x-1 hover:text-white transition-colors"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>Copy</span>
            </button>
          </div>
          <pre className="p-3 overflow-x-auto">
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
        return <p key={idx} className="whitespace-pre-wrap leading-relaxed">{p}</p>;
      }
      return p;
    });
  };

  return (
    <div className={`py-4 px-4 sm:px-6 transition-colors ${isUser ? 'bg-transparent' : 'bg-slate-100/50 dark:bg-darkbg-800/40 border-y border-slate-100 dark:border-slate-800'}`}>
      <div className="max-w-4xl mx-auto flex space-x-3.5">
        {/* Avatar */}
        <div className={`w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 text-white shadow-sm ${
          isUser 
            ? 'bg-gradient-to-br from-indigo-500 to-purple-600' 
            : 'bg-gradient-to-br from-brand-500 to-indigo-600 shadow-brand-500/20'
        }`}>
          {isUser ? <User className="w-4 h-4" /> : <Scale className="w-4 h-4" />}
        </div>

        {/* Message Content & Actions */}
        <div className="flex-1 min-w-0 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-700 dark:text-slate-200">
              {isUser ? 'You' : 'LawSLM Assistant'}
            </span>
            <span className="text-[10px] text-slate-400 flex items-center space-x-1">
              <Clock className="w-3 h-3" />
              <span>{message.timestamp}</span>
            </span>
          </div>

          {/* Body Text */}
          <div className="text-sm text-slate-800 dark:text-slate-200">
            {renderFormattedContent(message.content)}
            {message.isStreaming && (
              <span className="inline-block w-2 h-4 ml-1 bg-brand-500 animate-pulse" />
            )}
          </div>

          {/* Interactive PDF Document Card if available */}
          {message.pdfPreview && (
            <div className="mt-3 p-3 rounded-xl bg-brand-50/50 dark:bg-brand-500/10 border border-brand-200 dark:border-brand-500/30 flex items-center justify-between">
              <div className="flex items-center space-x-2.5">
                <FileText className="w-5 h-5 text-brand-500" />
                <div>
                  <h4 className="text-xs font-semibold text-brand-700 dark:text-brand-300">{message.pdfPreview.title}</h4>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400">Formal PDF Document Report Ready</p>
                </div>
              </div>
              <button
                onClick={() => onOpenPDFPreview && onOpenPDFPreview(message.pdfPreview!)}
                className="px-3 py-1.5 rounded-lg bg-brand-500 hover:bg-brand-600 text-white text-xs font-medium transition-colors shadow-sm"
              >
                Preview & Export PDF
              </button>
            </div>
          )}

          {/* Bottom Action Toolbar for Assistant */}
          {!isUser && !message.isStreaming && (
            <div className="pt-2 flex items-center justify-between text-slate-400 text-xs">
              <div className="flex items-center space-x-3">
                <button
                  onClick={handleCopy}
                  className="flex items-center space-x-1 hover:text-brand-500 transition-colors"
                  title="Copy Message"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copied ? 'Copied' : 'Copy'}</span>
                </button>

                <button
                  onClick={() => setLiked(liked === true ? null : true)}
                  className={`p-1 hover:text-emerald-500 transition-colors ${liked === true ? 'text-emerald-500' : ''}`}
                  title="Good Response"
                >
                  <ThumbsUp className="w-3.5 h-3.5" />
                </button>

                <button
                  onClick={() => setLiked(liked === false ? null : false)}
                  className={`p-1 hover:text-rose-500 transition-colors ${liked === false ? 'text-rose-500' : ''}`}
                  title="Bad Response"
                >
                  <ThumbsDown className="w-3.5 h-3.5" />
                </button>

                {onRegenerate && (
                  <button
                    onClick={onRegenerate}
                    className="flex items-center space-x-1 hover:text-brand-500 transition-colors"
                    title="Regenerate Response"
                  >
                    <RotateCw className="w-3.5 h-3.5" />
                    <span>Retry</span>
                  </button>
                )}
              </div>

              {message.responseTimeMs && (
                <div className="flex items-center space-x-1 text-[10px] text-slate-400">
                  <Cpu className="w-3 h-3 text-brand-400" />
                  <span>{(message.responseTimeMs / 1000).toFixed(2)}s</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
