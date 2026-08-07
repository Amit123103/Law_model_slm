import React, { useState } from 'react';
import { 
  Plus, 
  Search, 
  Pin, 
  MessageSquare, 
  Trash2, 
  Edit3, 
  Settings, 
  Activity, 
  PanelLeftClose, 
  PanelLeftOpen, 
  Check, 
  X,
  Folder
} from 'lucide-react';
import type { Conversation } from '../types/chat';

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
      className={`fixed md:static inset-y-0 left-0 z-30 w-[320px] glass-sidebar flex flex-col transition-all duration-300 transform ${
        open ? 'translate-x-0' : '-translate-x-full md:translate-x-0 md:w-16'
      }`}
    >
      {/* Top Header Actions */}
      <div className="p-4 flex items-center justify-between border-b border-slate-800">
        <button
          onClick={onNewChat}
          className={`flex-1 flex items-center justify-center space-x-2 py-3 px-4 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold text-sm shadow-md shadow-blue-500/25 transition-all ${
            !open && 'md:px-0 md:w-10'
          }`}
        >
          <Plus className="w-5 h-5" />
          <span className={`${!open && 'md:hidden'}`}>New Conversation</span>
        </button>

        <button
          onClick={() => setOpen(!open)}
          className="p-2.5 rounded-xl text-slate-400 hover:bg-slate-800 hidden md:block ml-2 transition-colors"
          title={open ? "Collapse Sidebar" : "Expand Sidebar"}
        >
          {open ? <PanelLeftClose className="w-5 h-5" /> : <PanelLeftOpen className="w-5 h-5" />}
        </button>
      </div>

      {/* Search Input */}
      {open && (
        <div className="px-4 pt-3 pb-1">
          <div className="relative">
            <Search className="w-4 h-4 absolute left-3.5 top-3 text-slate-400" />
            <input
              type="text"
              placeholder="Search conversations..."
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              className="w-full pl-9 pr-3.5 py-2 text-xs rounded-xl bg-slate-800/80 border border-slate-700/80 text-white placeholder-slate-400 outline-none focus:border-blue-500 transition-all"
            />
          </div>
        </div>
      )}

      {/* Conversation List */}
      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-5">
        {/* Pinned Chats */}
        {pinned.length > 0 && (
          <div>
            <div className={`px-2 text-[10px] font-extrabold tracking-wider text-slate-400 uppercase mb-2 ${!open && 'md:hidden'}`}>
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

        {/* Recent Conversations */}
        <div>
          <div className={`px-2 text-[10px] font-extrabold tracking-wider text-slate-400 uppercase mb-2 ${!open && 'md:hidden'}`}>
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

        {/* Document Folders Shortcut */}
        {open && (
          <div>
            <div className="px-2 text-[10px] font-extrabold tracking-wider text-slate-400 uppercase mb-2">
              Legal Folders
            </div>
            <div className="space-y-1">
              <div className="flex items-center space-x-2.5 px-3 py-2 rounded-xl text-xs text-slate-400 hover:bg-slate-800/60 cursor-pointer transition-colors">
                <Folder className="w-4 h-4 text-amber-400" />
                <span>Affidavits & Contracts</span>
              </div>
              <div className="flex items-center space-x-2.5 px-3 py-2 rounded-xl text-xs text-slate-400 hover:bg-slate-800/60 cursor-pointer transition-colors">
                <Folder className="w-4 h-4 text-blue-400" />
                <span>Python & PyTorch Scripts</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer System Nav */}
      <div className="p-3 border-t border-slate-800 space-y-2">
        <button
          onClick={onOpenDashboard}
          className="w-full flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-xs bg-slate-800/80 hover:bg-slate-700 border border-slate-700 text-slate-200 font-medium transition-all"
        >
          <Activity className="w-4 h-4 text-blue-400" />
          <span className={`${!open && 'md:hidden'}`}>Model Dashboard</span>
        </button>

        <button
          onClick={onOpenSettings}
          className="w-full flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-xs bg-slate-800/80 hover:bg-slate-700 border border-slate-700 text-slate-200 font-medium transition-all"
        >
          <Settings className="w-4 h-4 text-slate-400" />
          <span className={`${!open && 'md:hidden'}`}>Parameters & Settings</span>
        </button>

        {open && (
          <div className="pt-2 px-3 pb-1 border-t border-slate-800 flex items-center justify-between text-[11px] text-slate-400">
            <span className="font-semibold text-slate-300">LawSLM Engine</span>
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
      className={`group relative flex items-center justify-between px-3.5 py-2.5 rounded-xl text-xs cursor-pointer transition-all ${
        active
          ? 'bg-blue-600/20 border border-blue-500/40 text-blue-300 font-semibold'
          : 'text-slate-300 hover:bg-slate-800/80 border border-transparent'
      }`}
    >
      <div className="flex items-center space-x-3 min-w-0">
        <MessageSquare className="w-4 h-4 flex-shrink-0 text-slate-400 group-hover:text-blue-400 transition-colors" />
        {editing ? (
          <input
            type="text"
            value={editingTitle}
            onChange={e => setEditingTitle(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && onSaveRename()}
            className="w-full px-2 py-1 text-xs bg-slate-900 border border-blue-500 rounded-lg text-white outline-none"
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
              <button onClick={onSaveRename} className="p-1 hover:text-emerald-400"><Check className="w-3.5 h-3.5" /></button>
              <button onClick={onCancelRename} className="p-1 hover:text-rose-400"><X className="w-3.5 h-3.5" /></button>
            </>
          ) : (
            <>
              <button onClick={e => { e.stopPropagation(); onPin(); }} className="p-1 hover:text-amber-400">
                <Pin className={`w-3.5 h-3.5 ${conversation.pinned ? 'fill-amber-400 text-amber-400' : ''}`} />
              </button>
              <button onClick={e => { e.stopPropagation(); onStartRename(); }} className="p-1 hover:text-blue-400">
                <Edit3 className="w-3.5 h-3.5" />
              </button>
              <button onClick={e => { e.stopPropagation(); onDelete(); }} className="p-1 hover:text-rose-400">
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
};
