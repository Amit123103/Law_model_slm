import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { ChatArea } from './components/ChatArea';
import { ChatInput } from './components/ChatInput';
import { SettingsModal } from './components/SettingsModal';
import { PDFPreviewModal } from './components/PDFPreviewModal';
import { DashboardModal } from './components/DashboardModal';
import type { Conversation, Message, ModelParams, ModelStats } from './types/chat';
import { streamChatMessage, fetchModelInfo, fetchSystemPrompt } from './services/api';

const DEFAULT_PARAMS: ModelParams = {
  temperature: 0.0,
  topK: 1,
  topP: 1.0,
  maxTokens: 128,
  repetitionPenalty: 1.05,
  streaming: true
};

const INITIAL_CONVERSATION: Conversation = {
  id: 'conv-1',
  title: 'LawSLM Assistant Welcome Chat',
  pinned: true,
  updatedAt: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
  messages: [
    {
      id: 'msg-1',
      role: 'assistant',
      content: 'Hello! I am **LawSLM**, a Small Language Model built completely from scratch by **Amit Kumar**. I can assist you with legal information, programming, report generation, and general AI tasks. How can I help you today?',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]
};

export function App() {
  const [darkMode, setDarkMode] = useState<boolean>(() => {
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  });
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);
  const [conversations, setConversations] = useState<Conversation[]>([INITIAL_CONVERSATION]);
  const [activeId, setActiveId] = useState<string>('conv-1');
  const [params, setParams] = useState<ModelParams>(DEFAULT_PARAMS);
  const [systemPrompt, setSystemPrompt] = useState<string>('');
  const [modelStats, setModelStats] = useState<ModelStats>({
    status: 'online',
    vocabSize: 2000,
    dModel: 128,
    nLayers: 2,
    nHeads: 4,
    totalParams: '624.38K',
    trainableParams: '624.38K',
    activeDevice: 'CPU',
    ramUsage: '128 MB',
    checkpointLoaded: 'checkpoints/best_model.pt'
  });

  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [settingsOpen, setSettingsOpen] = useState<boolean>(false);
  const [dashboardOpen, setDashboardOpen] = useState<boolean>(false);
  const [pdfModalData, setPdfModalData] = useState<Message['pdfPreview'] | null>(null);
  const [systemPromptOpen, setSystemPromptOpen] = useState<boolean>(false);

  // Sync dark mode class with body
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  // Load API Stats & System Prompt on mount
  useEffect(() => {
    fetchModelInfo().then(stats => stats && setModelStats(stats));
    fetchSystemPrompt().then(prompt => setSystemPrompt(prompt));
  }, []);

  const activeConv = conversations.find(c => c.id === activeId) || conversations[0];

  const handleNewChat = () => {
    const newConv: Conversation = {
      id: `conv-${Date.now()}`,
      title: 'New Conversation',
      pinned: false,
      updatedAt: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      messages: []
    };
    setConversations([newConv, ...conversations]);
    setActiveId(newConv.id);
  };

  const handleDeleteChat = (id: string) => {
    const updated = conversations.filter(c => c.id !== id);
    if (updated.length === 0) {
      handleNewChat();
    } else {
      setConversations(updated);
      if (activeId === id) setActiveId(updated[0].id);
    }
  };

  const handleRenameChat = (id: string, newTitle: string) => {
    setConversations(conversations.map(c => c.id === id ? { ...c, title: newTitle } : c));
  };

  const handlePinChat = (id: string) => {
    setConversations(conversations.map(c => c.id === id ? { ...c, pinned: !c.pinned } : c));
  };

  const handleSendMessage = async (text: string, attachments?: any[]) => {
    if (!text.trim() && (!attachments || attachments.length === 0)) return;

    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const userMsg: Message = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp,
      attachments
    };

    const shouldRename = activeConv.messages.length === 0;
    const updatedTitle = shouldRename ? (text.length > 25 ? text.substring(0, 25) + '...' : text) : activeConv.title;

    const updatedMessages = [...activeConv.messages, userMsg];
    setConversations(conversations.map(c => c.id === activeId ? {
      ...c,
      title: updatedTitle,
      messages: updatedMessages,
      updatedAt: timestamp
    } : c));

    setIsGenerating(true);

    const assistantMsgId = `msg-${Date.now() + 1}`;
    const startTime = Date.now();

    const isPDFRequest = text.toLowerCase().includes('pdf') || text.toLowerCase().includes('report') || text.toLowerCase().includes('affidavit') || text.toLowerCase().includes('notice');
    
    let pdfPreviewData: Message['pdfPreview'] | undefined = undefined;
    if (isPDFRequest) {
      pdfPreviewData = {
        title: "LEGAL NOTICE & FORMAL DEMAND REPORT",
        summary: "Formal Legal Notice issued under statutory provisions demanding compliance and legal clarification.",
        sections: [
          { heading: "PARTIES & BACKGROUND", body: "This legal notice is served on behalf of the Claimant regarding unresolved contractual obligations." },
          { heading: "STATUTORY GROUNDS & SECTIONS", body: "Under relevant statutory codes, failure to comply within 15 days will result in formal court litigation." },
          { heading: "DEMAND & REMEDIES", body: "The Addressee is hereby directed to remit the outstanding sum and provide written confirmation." }
        ]
      };
    }

    const placeholderAssistantMsg: Message = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      isStreaming: true,
      pdfPreview: pdfPreviewData
    };

    setConversations(prev => prev.map(c => c.id === activeId ? {
      ...c,
      messages: [...c.messages, userMsg, placeholderAssistantMsg]
    } : c));

    try {
      let currentPdfMeta: Message['pdfPreview'] = pdfPreviewData;
      const responseText = await streamChatMessage(
        text,
        params,
        (chunkText: string) => {
          setConversations(prev => prev.map(c => {
            if (c.id !== activeId) return c;
            const msgs: Message[] = c.messages.map((m): Message => {
              if (m.id === assistantMsgId) {
                return { ...m, content: chunkText };
              }
              return m;
            });
            return { ...c, messages: msgs };
          }));
        },
        (meta: { hasPdf?: boolean; pdfMeta?: any }) => {
          if (meta.hasPdf && meta.pdfMeta) {
            currentPdfMeta = {
              title: meta.pdfMeta.title || "FORMAL DOCUMENT REPORT",
              summary: meta.pdfMeta.summary || "Generated by LawSLM",
              sections: [
                { heading: "EXECUTIVE SUMMARY", body: meta.pdfMeta.summary || "Official Document Summary" },
                { heading: "FULL DETAILS", body: meta.pdfMeta.content || "Report Details" }
              ]
            };
          }
        }
      );

      const endTime = Date.now();

      setConversations(prev => prev.map(c => {
        if (c.id !== activeId) return c;
        const msgs: Message[] = c.messages.map((m): Message => {
          if (m.id === assistantMsgId) {
            return {
              ...m,
              content: responseText || m.content,
              isStreaming: false,
              responseTimeMs: endTime - startTime,
              pdfPreview: currentPdfMeta
            };
          }
          return m;
        });
        return { ...c, messages: msgs };
      }));
    } catch (err) {
      console.error(err);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleRegenerate = () => {
    if (activeConv.messages.length < 2) return;
    const lastUserMsg = [...activeConv.messages].reverse().find(m => m.role === 'user');
    if (lastUserMsg) {
      handleSendMessage(lastUserMsg.content);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-50 dark:bg-darkbg-900 text-slate-900 dark:text-slate-100">
      {/* Sidebar Navigation */}
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onSelectConversation={setActiveId}
        onNewChat={handleNewChat}
        onDeleteChat={handleDeleteChat}
        onRenameChat={handleRenameChat}
        onPinChat={handlePinChat}
        onOpenSettings={() => setSettingsOpen(true)}
        onOpenDashboard={() => setDashboardOpen(true)}
        open={sidebarOpen}
        setOpen={setSidebarOpen}
      />

      {/* Main Workspace Column */}
      <div className="flex-1 flex flex-col min-w-0 h-full">
        <Header
          darkMode={darkMode}
          setDarkMode={setDarkMode}
          onOpenSettings={() => setSettingsOpen(true)}
          onOpenDashboard={() => setDashboardOpen(true)}
          onOpenSystemPrompt={() => setSystemPromptOpen(true)}
          modelStats={modelStats}
          sidebarOpen={sidebarOpen}
          setSidebarOpen={setSidebarOpen}
        />

        {/* Central Chat View */}
        <ChatArea
          messages={activeConv.messages}
          onSendMessage={handleSendMessage}
          onRegenerate={handleRegenerate}
          onOpenPDFPreview={pdf => setPdfModalData(pdf)}
        />

        {/* Chat Input Toolbar */}
        <ChatInput
          onSendMessage={handleSendMessage}
          isGenerating={isGenerating}
          onStopGeneration={() => setIsGenerating(false)}
        />
      </div>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        params={params}
        onUpdateParams={setParams}
        systemPrompt={systemPrompt}
        onUpdateSystemPrompt={setSystemPrompt}
      />

      {/* PDF Document Preview Modal */}
      <PDFPreviewModal
        isOpen={!!pdfModalData}
        onClose={() => setPdfModalData(null)}
        pdfData={pdfModalData || undefined}
      />

      {/* Model Dashboard Modal */}
      <DashboardModal
        isOpen={dashboardOpen}
        onClose={() => setDashboardOpen(false)}
        stats={modelStats}
      />

      {/* System Prompt View Drawer */}
      {systemPromptOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm">
          <div className="w-full max-w-xl glass-panel rounded-2xl p-6 shadow-2xl space-y-4">
            <div className="flex items-center justify-between border-b border-slate-200 dark:border-slate-800 pb-3">
              <h3 className="font-bold text-sm text-slate-900 dark:text-white">LawSLM System Prompt & Safety Guidelines</h3>
              <button onClick={() => setSystemPromptOpen(false)} className="text-slate-400 hover:text-slate-600">×</button>
            </div>
            <pre className="text-xs font-mono bg-slate-100 dark:bg-darkbg-900 p-4 rounded-xl max-h-96 overflow-y-auto whitespace-pre-wrap leading-relaxed text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-800">
              {systemPrompt}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
