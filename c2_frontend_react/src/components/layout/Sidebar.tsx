import React from 'react';
import { LayoutGrid, AlertOctagon, BarChart3, Settings, LucideIcon } from 'lucide-react';

interface SidebarProps {
  activeTab: 'live' | 'incidents' | 'analytics';
  onSelectTab: (tab: 'live' | 'incidents' | 'analytics') => void;
  unacknowledgedCount: number;
}

interface NavItem {
  id: 'live' | 'incidents' | 'analytics';
  label: string;
  icon: LucideIcon;
  badge?: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onSelectTab,
  unacknowledgedCount,
}) => {
  const navItems: NavItem[] = [
    { id: 'live', label: 'LIVE MATRIX', icon: LayoutGrid },
    { id: 'incidents', label: 'INCIDENT LOG', icon: AlertOctagon, badge: unacknowledgedCount },
    { id: 'analytics', label: 'ANALYTICS', icon: BarChart3 },
  ];

  return (
    <aside className="w-44 bg-[#0B0D13] border-r border-zinc-800/80 flex flex-col justify-between py-3 select-none">
      <div className="flex flex-col gap-1 px-2">
        <div className="px-3 py-1.5 text-[10px] font-mono font-bold text-zinc-500 uppercase tracking-widest">
          SYSTEM NAV
        </div>

        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;

          return (
            <button
              key={item.id}
              onClick={() => onSelectTab(item.id)}
              className={`flex items-center justify-between px-3 py-2 text-xs font-mono font-semibold rounded-xs border transition-colors ${
                isActive
                  ? 'bg-sky-950/80 text-sky-300 border-sky-800'
                  : 'text-zinc-400 border-transparent hover:bg-zinc-900 hover:text-zinc-200'
              }`}
            >
              <div className="flex items-center gap-2.5">
                <Icon className={`w-4 h-4 ${isActive ? 'text-sky-400' : 'text-zinc-500'}`} />
                <span>{item.label}</span>
              </div>

              {item.badge !== undefined && item.badge > 0 && (
                <span className="bg-rose-950 border border-rose-800 text-rose-400 text-[10px] px-1.5 py-0.2 rounded-xs font-bold animate-pulse">
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>

      <div className="px-2">
        <button
          className="w-full flex items-center gap-2.5 px-3 py-2 text-xs font-mono text-zinc-500 hover:text-zinc-300 transition-colors border border-transparent rounded-xs"
          onClick={() => alert('System Configuration Panel locked by administrator policy.')}
        >
          <Settings className="w-4 h-4 text-zinc-600" />
          <span>CONFIG</span>
        </button>
      </div>
    </aside>
  );
};
