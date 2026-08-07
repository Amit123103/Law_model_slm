import React, { useState } from 'react';
import { 
  Plus, 
  Search, 
  Pin, 
  MessageSquare, 
  Trash2, 
  Edit3, 
  Download, 
  FolderPlus, 
  Settings, 
  Activity, 
  HelpCircle, 
  PanelLeftClose, 
  PanelLeftOpen, 
  Check, 
  X,
  Scale
} from 'lucide-react';
import { Conversation } from '../types/chat';

interface SidebarProps {
  conversations: Conversation[];
  activeId: string;
  onSelectConversation: (id: string) => void;
  onNewChat: () => void;
  onDeleteChat: (id: string) => void;
  onRenameChat: (id: string, title: string) => void;
  onPinChat: (id: string) => void;
  onOpenSettings: () => void;
  onOpenDashboard: () => void;
  open: boolean;
  setOpen: (val: boolean) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  conversations,
  activeId,
  onSelectConversation,
  onNewChat,
  onDeleteChat,
  onRenameChat,
  onPinChat,
  onOpenSettings,
  onOpenDashboard,
  open,
  setOpen
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState('');

  const filtered = conversations.filter(c => 
    c.title.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const pinned = filtered.filter(c => c.pinned);
  const recent = filtered.filter(c => !c.pinned);

  const handleStartRename = (c: Conversation) => {
    setEditingId(c.id);
    setEditingTitle(c.title);
  };

  const handleSaveRename = (id: string) => {
    if (editingTitle.trim()) {
      onRenameChat(id, editingTitle.trim());
    }
    setEditingId(null);
  };

  return (
    <aside
      className={`fixed md:static inset-y-0 left-0 z-30 w-72 bg-white dark:bg-darkbg-800 border-r border-slate-200 dark:border-slate-800 flex flex-col transition-all duration-300 transform ${
        open ? 'translate-x-0' : '-translate-x-full md:translate-x-0 md:w-16'
      }`}
    >
      {/* Top Header Actions */}
      <div className="p-3 flex items-center justify-between border-b border-slate-100 dark:border-slate-750">
        <button
          onClick={onNewChat}
          className={`flex-1 flex items-center justify-center space-x-2 py-2.5 px-3 rounded-xl bg-brand-500 hover:bg-brand-600 text-white font-medium text-sm shadow-md shadow-brand-500/25 transition-all ${
            !open && 'md:px-0 md:w-10'
          }`}
        >
          <Plus className="w-4 h-4" />
          <span className={`${!open && 'md:hidden'}`}>New Chat</span>
        </button>

        <button
          onClick={() => setOpen(!open)}
          className="p-2 rounded-xl text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-700 hidden md:block ml-2"
          title={open ? "Collapse Sidebar" : "Expand Sidebar"}
        >
          {open ? <PanelLeftClose className="w-4 h-4" /> : <PanelLeftOpen className="w-4 h-4" />}
        </button>
      </div>

      {/* Search Input (visible when open) */}
      {open && (
        <div className="px-3 pt-3">
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-3 top-3 text-slate-400" />
            <input
              type="text"
              placeholder="Search conversations..."
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              className="w-full pl-8 pr-3 py-1.5 text-xs rounded-xl bg-slate-100 dark:bg-darkbg-900 border border-transparent focus:border-brand-500 focus:bg-white dark:focus:bg-darkbg-900 outline-none transition-colors"
            />
          </div>
        </div>
      )}

      {/* Conversation List */}
      <div className="flex-1 overflow-y-auto px-2 py-3 space-y-4">
        {/* Pinned Section */}
        {pinned.length > 0 && (
          <div>
            <div className={`px-2 text-[10px] font-bold tracking-wider text-slate-400 uppercase mb-1.5 ${!open && 'md:hidden'}`}>
              Pinned Chats
            </div>
            <div className="space-y-1">
              {pinned.map(c => (
                <ConversationItem
                  key={c.id}
                  conversation={c}
                  active={c.id === activeId}
                  open={open}
                  editing={editingId === c.id}
                  editingTitle={editingTitle}
                  setEditingTitle={setEditingTitle}
                  onSelect={() => onSelectConversation(c.id)}
                  onStartRename={() => handleStartRename(c)}
                  onSaveRename={() => handleSaveRename(c.id)}
                  onCancelRename={() => setEditingId(null)}
                  onDelete={() => onDeleteChat(c.id)}
                  onPin={() => onPinChat(c.id)}
                />
              ))}
            </div>
          </div>
        )}

        {/* Recent Section */}
        <div>
          <div className={`px-2 text-[10px] font-bold tracking-wider text-slate-400 uppercase mb-1.5 ${!open && 'md:hidden'}`}>
            Recent Conversations
          </div>
          <div className="space-y-1">
            {recent.map(c => (
              <ConversationItem
                key={c.id}
                conversation={c}
                active={c.id === activeId}
                open={open}
                editing={editingId === c.id}
                editingTitle={editingTitle}
                setEditingTitle={setEditingTitle}
                onSelect={() => onSelectConversation(c.id)}
                onStartRename={() => handleStartRename(c)}
                onSaveRename={() => handleSaveRename(c.id)}
                onCancelRename={() => setEditingId(null)}
                onDelete={() => onDeleteChat(c.id)}
                onPin={() => onPinChat(c.id)}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Footer System Nav */}
      <div className="p-2 border-t border-slate-100 dark:border-slate-750 space-y-1">
        <button
          onClick={onOpenDashboard}
          className="w-full flex items-center space-x-2.5 px-3 py-2 rounded-xl text-xs text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
        >
          <Activity className="w-4 h-4 text-brand-500" />
          <span className={`${!open && 'md:hidden'}`}>Model Dashboard</span>
        </button>

        <button
          onClick={onOpenSettings}
          className="w-full flex items-center space-x-2.5 px-3 py-2 rounded-xl text-xs text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
        >
          <Settings className="w-4 h-4 text-slate-500" />
          <span className={`${!open && 'md:hidden'}`}>Parameters & Settings</span>
        </button>

        {open && (
          <div className="pt-2 px-3 pb-1 border-t border-slate-100 dark:border-slate-750/60 flex items-center justify-between text-[11px] text-slate-400">
            <span>LawSLM Engine</span>
            <span>By Amit Kumar</span>
          </div>
        )}
      </div>
    </aside>
  );
};

interface ItemProps {
  conversation: Conversation;
  active: boolean;
  open: boolean;
  editing: boolean;
  editingTitle: string;
  setEditingTitle: (val: string) => void;
  onSelect: () => void;
  onStartRename: () => void;
  onSaveRename: () => void;
  onCancelRename: () => void;
  onDelete: () => void;
  onPin: () => void;
}

const ConversationItem: React.FC<ItemProps> = ({
  conversation,
  active,
  open,
  editing,
  editingTitle,
  setEditingTitle,
  onSelect,
  onStartRename,
  onSaveRename,
  onCancelRename,
  onDelete,
  onPin
}) => {
  return (
    <div
      onClick={onSelect}
      className={`group relative flex items-center justify-between px-3 py-2 rounded-xl text-xs cursor-pointer transition-colors ${
        active
          ? 'bg-brand-50 dark:bg-brand-500/15 text-brand-600 dark:text-brand-400 font-semibold'
          : 'text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-750'
      }`}
    >
      <div className="flex items-center space-x-2.5 min-w-0">
        <MessageSquare className="w-4 h-4 flex-shrink-0 text-slate-400 group-hover:text-brand-500" />
        {editing ? (
          <input
            type="text"
            value={editingTitle}
            onChange={e => setEditingTitle(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && onSaveRename()}
            className="w-full px-1.5 py-0.5 text-xs bg-white dark:bg-darkbg-900 border border-brand-500 rounded outline-none"
            autoFocus
          />
        ) : (
          <span className={`truncate ${!open && 'md:hidden'}`}>{conversation.title}</span>
        )}
      </div>

      {open && (
        <div className="flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
          {editing ? (
            <>
              <button onClick={onSaveRename} className="p-1 hover:text-emerald-500"><Check className="w-3.5 h-3.5" /></button>
              <button onClick={onCancelRename} className="p-1 hover:text-rose-500"><X className="w-3.5 h-3.5" /></button>
            </>
          ) : (
            <>
              <button onClick={e => { e.stopPropagation(); onPin(); }} className="p-1 hover:text-amber-500">
                <Pin className={`w-3.5 h-3.5 ${conversation.pinned ? 'fill-amber-500 text-amber-500' : ''}`} />
              </button>
              <button onClick={e => { e.stopPropagation(); onStartRename(); }} className="p-1 hover:text-brand-500">
                <Edit3 className="w-3.5 h-3.5" />
              </button>
              <button onClick={e => { e.stopPropagation(); onDelete(); }} className="p-1 hover:text-rose-500">
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
};
