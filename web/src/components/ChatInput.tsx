import React, { useState, useRef } from 'react';
import type { KeyboardEvent } from 'react';
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
      name: type === 'pdf' ? 'Legal_Affidavit_Draft.pdf' : 'Sample_Document.png',
      size: type === 'pdf' ? '240 KB' : '1.2 MB',
      type: type
    };
    setAttachments([...attachments, fakeFile]);
    setShowAttachMenu(false);
  };

  return (
    <div className="p-3 sm:p-4 max-w-4xl mx-auto w-full shrink-0">
      {/* Attached Files Pill Preview */}
      {attachments.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-2">
          {attachments.map((att, idx) => (
            <div key={idx} className="flex items-center space-x-2 px-3 py-1 rounded-xl bg-blue-500/15 border border-blue-500/30 text-xs text-blue-300">
              <FileText className="w-3.5 h-3.5" />
              <span className="font-medium">{att.name}</span>
              <span className="text-[10px] text-slate-400">({att.size})</span>
              <button 
                onClick={() => setAttachments(attachments.filter((_, i) => i !== idx))}
                className="hover:text-rose-400 ml-1 font-bold"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="relative rounded-2xl p-2.5 bg-slate-800/90 border border-slate-700 shadow-xl">
        {/* Text Input Area */}
        <textarea
          ref={textareaRef}
          value={text}
          onChange={handleTextChange}
          onKeyDown={handleKeyDown}
          placeholder="Ask LawSLM anything (e.g. 'Explain laws in simple terms' or 'Create PDF report')..."
          rows={1}
          className="w-full px-3 py-2 text-sm bg-slate-900 border border-slate-700 rounded-xl outline-none resize-none text-white placeholder-slate-400 focus:border-blue-500"
        />

        {/* Bottom Toolbar & Action Controls */}
        <div className="flex items-center justify-between pt-2 px-1">
          <div className="relative flex items-center space-x-1.5">
            {/* Attachment Button */}
            <button
              type="button"
              onClick={() => setShowAttachMenu(!showAttachMenu)}
              className="p-2 rounded-xl bg-slate-700/80 hover:bg-slate-700 text-slate-300 transition-colors"
              title="Attach Document / Image"
            >
              <Paperclip className="w-4 h-4" />
            </button>

            {/* Voice Input Trigger */}
            <button
              type="button"
              onClick={() => setText("Explain Section 420 in simple language")}
              className="p-2 rounded-xl bg-slate-700/80 hover:bg-slate-700 text-slate-300 transition-colors"
              title="Voice Prompt Sample"
            >
              <Mic className="w-4 h-4" />
            </button>

            {/* Quick Prompt Suggestion */}
            <button
              type="button"
              onClick={() => setText("Generate formal legal notice PDF report")}
              className="px-3 py-1.5 rounded-xl bg-amber-500/10 text-amber-400 border border-amber-500/30 text-xs font-medium hidden sm:flex items-center space-x-1 hover:bg-amber-500/20 transition-colors"
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>Generate PDF Prompt</span>
            </button>

            {/* Attachment Menu Drawer */}
            {showAttachMenu && (
              <div className="absolute left-0 bottom-12 z-20 w-52 p-2 bg-slate-800 rounded-2xl shadow-2xl border border-slate-700 space-y-1">
                <button
                  onClick={() => handleAttachFile('pdf')}
                  className="w-full flex items-center space-x-2.5 px-3 py-2 rounded-xl text-xs hover:bg-slate-700 text-slate-200"
                >
                  <FileText className="w-4 h-4 text-rose-400" />
                  <span>Upload PDF Document</span>
                </button>
                <button
                  onClick={() => handleAttachFile('image')}
                  className="w-full flex items-center space-x-2.5 px-3 py-2 rounded-xl text-xs hover:bg-slate-700 text-slate-200"
                >
                  <ImageIcon className="w-4 h-4 text-blue-400" />
                  <span>Upload Vision Image</span>
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
                className="flex items-center space-x-1.5 px-3.5 py-2 rounded-xl bg-rose-600 hover:bg-rose-700 text-white text-xs font-semibold shadow-md transition-all"
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
                    ? 'bg-blue-600 hover:bg-blue-700 shadow-blue-500/25'
                    : 'bg-slate-700 text-slate-500 cursor-not-allowed'
                }`}
              >
                <Send className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </div>

      <p className="text-[11px] text-center text-slate-400 mt-2">
        LawSLM can assist with legal info, code & general tasks. Verify important legal advice with a qualified lawyer.
      </p>
    </div>
  );
};
