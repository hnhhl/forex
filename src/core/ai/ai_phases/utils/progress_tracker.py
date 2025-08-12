"""
Progress Tracker Module

Module này theo dõi tiến độ phát triển của tất cả 6 phases.
"""

from datetime import datetime
import json

class PhaseProgressTracker:
    """📊 Track development progress of all 6 phases"""
    
    def __init__(self):
        self.phases_status = {
            'Phase 1': {'name': 'Online Learning Engine', 'boost': 2.5, 'status': 'COMPLETED', 'progress': 100},
            'Phase 2': {'name': 'Advanced Backtest Framework', 'boost': 1.5, 'status': 'COMPLETED', 'progress': 100},
            'Phase 3': {'name': 'Adaptive Intelligence', 'boost': 3.0, 'status': 'COMPLETED', 'progress': 100},
            'Phase 4': {'name': 'Multi-Market Learning', 'boost': 2.0, 'status': 'COMPLETED', 'progress': 100},
            'Phase 5': {'name': 'Real-Time Enhancement', 'boost': 1.5, 'status': 'COMPLETED', 'progress': 100},
            'Phase 6': {'name': 'Future Evolution', 'boost': 1.5, 'status': 'COMPLETED', 'progress': 100}
        }
        
        self.development_log = []
        
    def generate_progress_report(self):
        """Generate comprehensive progress report"""
        total_boost_target = sum(phase['boost'] for phase in self.phases_status.values())
        completed_boost = sum(phase['boost'] for phase in self.phases_status.values() 
                            if phase['status'] == 'COMPLETED')
        
        overall_progress = sum(phase['progress'] for phase in self.phases_status.values()) / 6
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_progress': f"{overall_progress:.1f}%",
            'total_boost_target': f"+{total_boost_target}%",
            'completed_boost': f"+{completed_boost}%",
            'phases_detail': self.phases_status
        }
    
    def update_phase_progress(self, phase_number, progress, status=None):
        """Update progress for a specific phase
        
        Args:
            phase_number (int): Phase number (1-6)
            progress (float): Progress percentage (0-100)
            status (str, optional): Status ('PENDING', 'IN_PROGRESS', 'COMPLETED')
        """
        phase_key = f'Phase {phase_number}'
        if phase_key in self.phases_status:
            self.phases_status[phase_key]['progress'] = progress
            
            if status:
                self.phases_status[phase_key]['status'] = status
            
            # Auto-update status based on progress
            if progress >= 100 and not status:
                self.phases_status[phase_key]['status'] = 'COMPLETED'
            elif progress > 0 and progress < 100 and not status:
                self.phases_status[phase_key]['status'] = 'IN_PROGRESS'
            
            # Add to development log
            self.development_log.append({
                'timestamp': datetime.now().isoformat(),
                'phase': phase_key,
                'progress': progress,
                'status': self.phases_status[phase_key]['status']
            })
    
    def get_phase_status(self, phase_number):
        """Get status for a specific phase
        
        Args:
            phase_number (int): Phase number (1-6)
            
        Returns:
            dict: Phase status information
        """
        phase_key = f'Phase {phase_number}'
        return self.phases_status.get(phase_key, {})
    
    def print_progress_report(self):
        """Print formatted progress report to console"""
        report = self.generate_progress_report()
        print("\n📊 DEVELOPMENT PROGRESS REPORT:")
        print(json.dumps(report, indent=2, default=str))
        
    def reset_progress(self):
        """Reset all phases to PENDING with 0% progress"""
        for phase_key in self.phases_status:
            self.phases_status[phase_key]['status'] = 'PENDING'
            self.phases_status[phase_key]['progress'] = 0
        
        self.development_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'reset',
            'message': 'All phases reset to PENDING'
        }) 