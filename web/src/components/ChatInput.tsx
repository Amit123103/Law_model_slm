import React, { useState, useRef, KeyboardEvent } from 'react';
import { Send, Paperclip, Mic, FileText, Image as ImageIcon, Sparkles, StopCircle } from 'lucide-react';

interface ChatInputProps {
  onSendMessage: (text: string, attachments?: any[]) => void;
  isGenerating: boolean;
  onStopGeneration?: () => void;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  isGenerating,
  onStopGeneration
}) => {
  const [text, setText] = useState('');
  const [showAttachMenu, setShowAttachMenu] = useState(false);
  const [attachments, setAttachments] = useState<{ name: string; size: string; type: string }[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    if ((!text.trim() && attachments.length === 0) || isGenerating) return;
    onSendMessage(text.trim(), attachments);
    setText('');
    setAttachments([]);
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
    }
  };

  const handleAttachFile = (type: string) => {
    const fakeFile = {
      name: type === 'pdf' ? 'Legal_Affidavit_Draft.pdf' : 'Sample_Code_Document.py',
      size: type === 'pdf' ? '240 KB' : '12 KB',
      type: type
    };
    setAttachments([...attachments, fakeFile]);
    setShowAttachMenu(false);
  };

  return (
    <div className="p-3 sm:p-4 max-w-4xl mx-auto w-full">
      {/* Attached Files Pill Preview */}
      {attachments.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-2">
          {attachments.map((att, idx) => (
            <div key={idx} className="flex items-center space-x-2 px-3 py-1 rounded-xl bg-brand-50 dark:bg-brand-500/15 border border-brand-200 dark:border-brand-500/30 text-xs text-brand-700 dark:text-brand-300">
              <FileText className="w-3.5 h-3.5" />
              <span className="font-medium">{att.name}</span>
              <span className="text-[10px] text-slate-400">({att.size})</span>
              <button 
                onClick={() => setAttachments(attachments.filter((_, i) => i !== idx))}
                className="hover:text-rose-500 ml-1 font-bold"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="relative glass-panel rounded-2xl p-2 shadow-lg border border-slate-200/80 dark:border-slate-800">
        {/* Text Input Area */}
        <textarea
          ref={textareaRef}
          value={text}
          onChange={handleTextChange}
          onKeyDown={handleKeyDown}
          placeholder="Ask LawSLM anything (e.g. 'Explain laws in simple terms' or 'Create PDF report')..."
          rows={1}
          className="w-full px-3 py-2 text-sm bg-transparent outline-none resize-none text-slate-800 dark:text-slate-100 placeholder-slate-400"
        />

        {/* Bottom Toolbar & Action Controls */}
        <div className="flex items-center justify-between pt-1 border-t border-slate-100 dark:border-slate-800/60 px-1">
          <div className="relative flex items-center space-x-1">
            {/* Attachment Button */}
            <button
              type="button"
              onClick={() => setShowAttachMenu(!showAttachMenu)}
              className="p-2 rounded-xl text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
              title="Attach Document / File"
            >
              <Paperclip className="w-4 h-4" />
            </button>

            {/* Voice Input Trigger */}
            <button
              type="button"
              onClick={() => setText("Explain Section 420 in simple language")}
              className="p-2 rounded-xl text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
              title="Voice Prompt Sample"
            >
              <Mic className="w-4 h-4" />
            </button>

            {/* Quick Prompt Suggestion */}
            <button
              type="button"
              onClick={() => setText("Generate formal legal notice PDF report")}
              className="px-2.5 py-1 rounded-xl bg-amber-50 dark:bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-200 dark:border-amber-500/30 text-[11px] font-medium hidden sm:flex items-center space-x-1 hover:bg-amber-100 transition-colors"
            >
              <Sparkles className="w-3 h-3" />
              <span>Generate PDF Prompt</span>
            </button>

            {/* Attachment Menu Drawer */}
            {showAttachMenu && (
              <div className="absolute left-0 bottom-10 z-20 w-48 p-2 glass-panel rounded-2xl shadow-xl border border-slate-200 dark:border-slate-700 space-y-1">
                <button
                  onClick={() => handleAttachFile('pdf')}
                  className="w-full flex items-center space-x-2 px-3 py-2 rounded-xl text-xs hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200"
                >
                  <FileText className="w-4 h-4 text-rose-500" />
                  <span>Upload PDF Document</span>
                </button>
                <button
                  onClick={() => handleAttachFile('image')}
                  className="w-full flex items-center space-x-2 px-3 py-2 rounded-xl text-xs hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200"
                >
                  <ImageIcon className="w-4 h-4 text-brand-500" />
                  <span>Upload Image</span>
                </button>
              </div>
            )}
          </div>

          <div className="flex items-center space-x-3">
            <span className="text-[10px] text-slate-400 hidden sm:inline">
              {text.length} chars
            </span>

            {isGenerating ? (
              <button
                type="button"
                onClick={onStopGeneration}
                className="flex items-center space-x-1.5 px-3 py-2 rounded-xl bg-rose-500 hover:bg-rose-600 text-white text-xs font-semibold shadow-md transition-all"
              >
                <StopCircle className="w-4 h-4" />
                <span>Stop</span>
              </button>
            ) : (
              <button
                type="button"
                onClick={handleSend}
                disabled={!text.trim() && attachments.length === 0}
                className={`p-2.5 rounded-xl text-white shadow-md transition-all ${
                  text.trim() || attachments.length > 0
                    ? 'bg-brand-500 hover:bg-brand-600 shadow-brand-500/25'
                    : 'bg-slate-300 dark:bg-slate-800 text-slate-400 cursor-not-allowed'
                }`}
              >
                <Send className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </div>

      <p className="text-[11px] text-center text-slate-400 dark:text-slate-500 mt-2">
        LawSLM can assist with legal info, code & general tasks. Verify important legal advice with a qualified lawyer.
      </p>
    </div>
  );
};
