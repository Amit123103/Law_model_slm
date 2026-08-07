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
  X
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
      className={`fixed md:static inset-y-0 left-0 z-30 w-72 bg-slate-900 border-r border-slate-800 flex flex-col transition-all duration-300 transform ${
        open ? 'translate-x-0' : '-translate-x-full md:translate-x-0 md:w-16'
      }`}
    >
      {/* Top Header Actions */}
      <div className="p-3 flex items-center justify-between border-b border-slate-800">
        <button
          onClick={onNewChat}
          className={`flex-1 flex items-center justify-center space-x-2 py-2.5 px-3 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-medium text-sm shadow-md shadow-blue-500/20 transition-all ${
            !open && 'md:px-0 md:w-10'
          }`}
        >
          <Plus className="w-4 h-4" />
          <span className={`${!open && 'md:hidden'}`}>New Chat</span>
        </button>

        <button
          onClick={() => setOpen(!open)}
          className="p-2 rounded-xl text-slate-400 hover:bg-slate-800 hidden md:block ml-2"
          title={open ? "Collapse Sidebar" : "Expand Sidebar"}
        >
          {open ? <PanelLeftClose className="w-4 h-4" /> : <PanelLeftOpen className="w-4 h-4" />}
        </button>
      </div>

      {/* Search Input */}
      {open && (
        <div className="px-3 pt-3">
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-3 top-3 text-slate-400" />
            <input
              type="text"
              placeholder="Search conversations..."
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              className="w-full pl-8 pr-3 py-1.5 text-xs rounded-xl bg-slate-800 border border-slate-700 text-white placeholder-slate-400 outline-none focus:border-blue-500"
            />
          </div>
        </div>
      )}

      {/* Conversation List */}
      <div className="flex-1 overflow-y-auto px-2 py-3 space-y-4">
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
      <div className="p-2 border-t border-slate-800 space-y-1.5">
        <button
          onClick={onOpenDashboard}
          className="w-full flex items-center space-x-2.5 px-3 py-2 rounded-xl text-xs bg-slate-800/80 hover:bg-slate-700 border border-slate-700 text-slate-200 transition-colors"
        >
          <Activity className="w-4 h-4 text-blue-400" />
          <span className={`${!open && 'md:hidden'}`}>Model Dashboard</span>
        </button>

        <button
          onClick={onOpenSettings}
          className="w-full flex items-center space-x-2.5 px-3 py-2 rounded-xl text-xs bg-slate-800/80 hover:bg-slate-700 border border-slate-700 text-slate-200 transition-colors"
        >
          <Settings className="w-4 h-4 text-slate-400" />
          <span className={`${!open && 'md:hidden'}`}>Parameters & Settings</span>
        </button>

        {open && (
          <div className="pt-2 px-3 pb-1 border-t border-slate-800 flex items-center justify-between text-[11px] text-slate-400">
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
          ? 'bg-blue-600/20 border border-blue-500/40 text-blue-400 font-semibold'
          : 'text-slate-300 hover:bg-slate-800 border border-transparent'
      }`}
    >
      <div className="flex items-center space-x-2.5 min-w-0">
        <MessageSquare className="w-4 h-4 flex-shrink-0 text-slate-400 group-hover:text-blue-400" />
        {editing ? (
          <input
            type="text"
            value={editingTitle}
            onChange={e => setEditingTitle(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && onSaveRename()}
            className="w-full px-1.5 py-0.5 text-xs bg-slate-900 border border-blue-500 rounded text-white outline-none"
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
