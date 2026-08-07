export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  responseTimeMs?: number;
  tokenCount?: number;
  isStreaming?: boolean;
  attachments?: {
    name: string;
    size: string;
    type: string;
  }[];
  pdfPreview?: {
    title: string;
    summary: string;
    sections: { heading: string; body: string }[];
  };
}

export interface Conversation {
  id: string;
  title: string;
  pinned: boolean;
  folderId?: string;
  messages: Message[];
  updatedAt: string;
}

export interface Folder {
  id: string;
  name: string;
  color?: string;
}

export interface ModelParams {
  temperature: number;
  topK: number;
  topP: number;
  maxTokens: number;
  repetitionPenalty: number;
  streaming: boolean;
}

export interface ModelStats {
  status: 'online' | 'busy' | 'offline';
  vocabSize: number;
  dModel: number;
  nLayers: number;
  nHeads: number;
  totalParams: string;
  trainableParams: string;
  activeDevice: string;
  ramUsage: string;
  checkpointLoaded: string;
}
