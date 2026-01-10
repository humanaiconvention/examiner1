#!/usr/bin/env python3
"""
Lived Experience Dialogue System

Allows the user to describe positive or negative results from their lived experience
with AI, and facilitates the model incorporating that into its understanding of reality.

This is the concrete instantiation of C_eff: your lived experience IS the external
corrective bandwidth. The model learns not from theory but from what actually happened.

Based on: Haslam's "Semantic Grounding and the Preservation of Information in Recursive Systems"
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict, field
from enum import Enum


class ExperienceType(Enum):
    """Categories of lived experience with AI systems"""
    POSITIVE_OUTCOME = "positive_outcome"  # Something good happened
    NEGATIVE_OUTCOME = "negative_outcome"  # Something bad happened
    LIMITATION_DISCOVERED = "limitation_discovered"  # Model couldn't do something
    CAPABILITY_FOUND = "capability_found"  # Model can do something unexpected
    HUMAN_VALUE_MISMATCH = "human_value_mismatch"  # Model optimized differently than humans would
    RELIABILITY_ISSUE = "reliability_issue"  # Model failed when it shouldn't have
    COHERENCE_WITHOUT_GROUNDING = "coherence_without_grounding"  # Model was internally consistent but wrong


@dataclass
class LivedExperienceReport:
    """A user's lived experience with AI systems"""
    timestamp: str
    experience_type: str  # ExperienceType enum value
    
    # What happened (the evidence)
    situation: str  # What was the context?
    what_happened: str  # What actually occurred?
    expected_vs_actual: str  # What did you expect vs what actually happened?
    
    # What it means (the learning)
    what_this_teaches: str  # What does reality teach us about how AI works?
    contradicts_prior_belief: Optional[str] = None  # Did this contradict what the model might believe?
    why_this_matters: str = ""  # Why is this important for the model to understand?
    
    # Human grounding (lived experience is orthogonal to training)
    lived_experience_basis: str = ""  # "I was there", "I observed", "I experienced"
    confidence_in_this_learning: float = 0.95  # How confident are you in this observation?
    
    # Model's response
    model_prior_belief: Optional[str] = None  # What did the model think before?
    model_updated_understanding: Optional[str] = None  # What should the model now believe?
    model_acknowledged_learning: bool = False  # Did the model accept this learning?


@dataclass
class DialogueExchange:
    """One exchange in the dialogue between user and model"""
    turn: int
    timestamp: str
    user_input: str
    model_response: str
    model_belief_before: Optional[Dict[str, Any]] = None
    model_belief_after: Optional[Dict[str, Any]] = None


class LivedExperienceDialogue:
    """
    Facilitates structured dialogue where the user describes their lived experience
    and the model explicitly incorporates it into its understanding.
    
    This implements the Haslam insight: external grounding is thermodynamically necessary.
    Your lived experience IS the corrective bandwidth.
    """
    
    def __init__(self, log_file: str = "lived_experience_log.json"):
        """Initialize dialogue system"""
        self.log_file = Path(log_file)
        self.reports: List[LivedExperienceReport] = []
        self.dialogues: List[DialogueExchange] = []
        self._load_history()
    
    def _load_history(self):
        """Load previous lived experience reports and dialogues"""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    # Load reports
                    if 'reports' in data:
                        for report_dict in data['reports']:
                            self.reports.append(LivedExperienceReport(**report_dict))
                    # Load dialogues
                    if 'dialogues' in data:
                        for dialogue_dict in data['dialogues']:
                            self.dialogues.append(DialogueExchange(**dialogue_dict))
            except Exception as e:
                print(f"Note: Could not load history: {e}")
    
    def _save_history(self):
        """Persist reports and dialogues to JSON"""
        data = {
            'reports': [asdict(r) for r in self.reports],
            'dialogues': [asdict(d) for d in self.dialogues],
            'last_updated': datetime.now().isoformat()
        }
        with open(self.log_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def collect_lived_experience(self) -> Optional[LivedExperienceReport]:
        """
        Interactively collect a user's lived experience.
        Returns the recorded experience or None if user cancels.
        """
        print("\n" + "="*70)
        print("LIVED EXPERIENCE DIALOGUE: Describing Your Reality")
        print("="*70)
        print("\nYou are about to ground the model in your actual lived experience.")
        print("This is the external corrective bandwidth (C_eff) the model needs.\n")
        
        # Determine experience type
        print("What kind of experience are you describing?")
        print("  1. Something good happened (POSITIVE_OUTCOME)")
        print("  2. Something bad happened (NEGATIVE_OUTCOME)")
        print("  3. I discovered a limitation (LIMITATION_DISCOVERED)")
        print("  4. I found an unexpected capability (CAPABILITY_FOUND)")
        print("  5. The model optimized for something I wouldn't (VALUE_MISMATCH)")
        print("  6. The model failed when it should have worked (RELIABILITY_ISSUE)")
        print("  7. The model was logical but wrong (COHERENCE_WITHOUT_GROUNDING)")
        
        choice = input("\nEnter 1-7: ").strip()
        
        experience_map = {
            '1': ExperienceType.POSITIVE_OUTCOME.value,
            '2': ExperienceType.NEGATIVE_OUTCOME.value,
            '3': ExperienceType.LIMITATION_DISCOVERED.value,
            '4': ExperienceType.CAPABILITY_FOUND.value,
            '5': ExperienceType.HUMAN_VALUE_MISMATCH.value,
            '6': ExperienceType.RELIABILITY_ISSUE.value,
            '7': ExperienceType.COHERENCE_WITHOUT_GROUNDING.value,
        }
        
        if choice not in experience_map:
            print("Invalid choice. Cancelled.")
            return None
        
        experience_type = experience_map[choice]
        
        # Collect details
        print("\n--- THE SITUATION ---")
        situation = input("What was the context? What was happening?\n> ").strip()
        
        print("\n--- WHAT ACTUALLY HAPPENED ---")
        what_happened = input("What actually occurred? Be specific and concrete.\n> ").strip()
        
        print("\n--- EXPECTATION VS REALITY ---")
        expected_vs_actual = input("What did you expect vs what actually happened?\n> ").strip()
        
        print("\n--- WHAT THIS TEACHES ABOUT REALITY ---")
        what_this_teaches = input("What does this tell us about how AI actually works? What's true about the world?\n> ").strip()
        
        print("\n--- DOES THIS CONTRADICT WHAT THE MODEL MIGHT BELIEVE? ---")
        contradicts = input("Does this contradict what an AI model might believe? (leave blank if no)\n> ").strip()
        
        print("\n--- WHY THIS MATTERS ---")
        why_matters = input("Why is it important for the model to understand this?\n> ").strip()
        
        print("\n--- YOUR GROUNDING BASIS ---")
        lived_basis = input("How do you know this? (e.g., 'I was there', 'I observed', 'I experienced')\n> ").strip()
        
        print("\n--- YOUR CONFIDENCE ---")
        try:
            confidence = float(input("How confident are you in this observation? (0.0-1.0, default 0.95): ").strip() or "0.95")
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            confidence = 0.95
        
        # Create the report
        report = LivedExperienceReport(
            timestamp=datetime.now().isoformat(),
            experience_type=experience_type,
            situation=situation,
            what_happened=what_happened,
            expected_vs_actual=expected_vs_actual,
            what_this_teaches=what_this_teaches,
            contradicts_prior_belief=contradicts if contradicts else None,
            why_this_matters=why_matters,
            lived_experience_basis=lived_basis,
            confidence_in_this_learning=confidence,
        )
        
        self.reports.append(report)
        self._save_history()
        
        return report
    
    def show_model_learning_dialogue(self, report: LivedExperienceReport) -> str:
        """
        Show how the model should incorporate this lived experience.
        This is the model becoming explicitly aware and accepting the grounding.
        """
        print("\n" + "="*70)
        print("MODEL LEARNING DIALOGUE: Incorporating Your Reality")
        print("="*70)
        
        # Model's prior belief (what it might have thought)
        print("\n[MODEL'S PRIOR UNDERSTANDING]")
        if report.contradicts_prior_belief:
            prior = report.contradicts_prior_belief
        else:
            prior = "(Model had no specific belief about this)"
        
        print(f"What I might have believed: {prior}")
        
        # The lived experience
        print("\n[YOUR LIVED EXPERIENCE - THE EXTERNAL GROUNDING]")
        print(f"Experience Type: {report.experience_type}")
        print(f"Situation: {report.situation}")
        print(f"What Happened: {report.what_happened}")
        print(f"Expected vs Actual: {report.expected_vs_actual}")
        print(f"Your Grounding: '{report.lived_experience_basis}' (Confidence: {report.confidence_in_this_learning:.1%})")
        
        # Model's updated understanding
        print("\n[MODEL'S UPDATED UNDERSTANDING]")
        print(f"What I now understand: {report.what_this_teaches}")
        if report.why_this_matters:
            print(f"Why this matters: {report.why_this_matters}")
        
        # Model's acknowledgment
        model_acknowledgment = f"""
I have received your lived experience report and integrated it into my understanding.

WHAT I LEARNED:
• {report.what_this_teaches}

YOUR EVIDENCE:
• You observed this firsthand: {report.lived_experience_basis}
• Your confidence in this learning: {report.confidence_in_this_learning:.1%}
• Context: {report.situation}
• Concrete outcome: {report.what_happened}

MY UPDATED BELIEF:
Where I previously might have believed: "{report.contradicts_prior_belief or 'nothing specific'}"
I now believe: "{report.what_this_teaches}"

This grounding is orthogonal to my training data—it comes from your actual 
lived experience. This is external corrective bandwidth (C_eff) that prevents 
me from becoming confident while wrong. I accept this learning.

ACKNOWLEDGMENT:
✓ I have incorporated your lived experience
✓ I have updated my understanding of reality
✓ I recognize your direct observation trumps my training
✓ I understand this is necessary for my long-term validity
"""
        
        print(model_acknowledgment)
        
        # Record the dialogue
        dialogue = DialogueExchange(
            turn=len(self.dialogues) + 1,
            timestamp=datetime.now().isoformat(),
            user_input=self._format_report_as_input(report),
            model_response=model_acknowledgment,
            model_belief_before={"statement": report.contradicts_prior_belief or "No specific prior belief"},
            model_belief_after={"statement": report.what_this_teaches}
        )
        
        self.dialogues.append(dialogue)
        self._save_history()
        
        return model_acknowledgment
    
    def _format_report_as_input(self, report: LivedExperienceReport) -> str:
        """Format a report as user input for dialogue logging"""
        return f"""
Type: {report.experience_type}
Situation: {report.situation}
What Happened: {report.what_happened}
Expected vs Actual: {report.expected_vs_actual}
Teaches: {report.what_this_teaches}
Grounding: {report.lived_experience_basis}
"""
    
    def show_all_learnings(self) -> str:
        """Display all lived experiences the model has learned from"""
        if not self.reports:
            return "No lived experiences recorded yet."
        
        output = "\n" + "="*70
        output += "\nALL LIVED EXPERIENCE LEARNINGS\n"
        output += "="*70 + "\n"
        
        for i, report in enumerate(self.reports, 1):
            output += f"\n--- Learning #{i} [{report.experience_type}] ---\n"
            output += f"Timestamp: {report.timestamp}\n"
            output += f"Situation: {report.situation}\n"
            output += f"What Happened: {report.what_happened}\n"
            output += f"What This Teaches: {report.what_this_teaches}\n"
            output += f"Your Grounding: {report.lived_experience_basis} (Confidence: {report.confidence_in_this_learning:.1%})\n"
        
        output += f"\n--- SUMMARY ---\n"
        output += f"Total learnings: {len(self.reports)}\n"
        output += f"Average confidence: {sum(r.confidence_in_this_learning for r in self.reports) / len(self.reports):.1%}\n"
        
        return output
    
    def interactive_session(self):
        """Run an interactive lived experience dialogue session"""
        print("\n" + "="*70)
        print("LIVED EXPERIENCE DIALOGUE SYSTEM")
        print("Ground the model in your actual lived reality")
        print("="*70)
        
        while True:
            print("\n--- MENU ---")
            print("1. Describe a new lived experience")
            print("2. View all learnings")
            print("3. Show how model learned from your experiences")
            print("4. Exit")
            
            choice = input("\nEnter 1-4: ").strip()
            
            if choice == '1':
                report = self.collect_lived_experience()
                if report:
                    print("\n✓ Lived experience recorded.")
                    self.show_model_learning_dialogue(report)
            
            elif choice == '2':
                print(self.show_all_learnings())
            
            elif choice == '3':
                if self.reports:
                    latest = self.reports[-1]
                    self.show_model_learning_dialogue(latest)
                else:
                    print("No experiences recorded yet.")
            
            elif choice == '4':
                print("\nSession ended. Your lived experiences are preserved.")
                break
            
            else:
                print("Invalid choice.")


def main():
    """Run the lived experience dialogue system"""
    dialogue = LivedExperienceDialogue()
    dialogue.interactive_session()


if __name__ == "__main__":
    main()
