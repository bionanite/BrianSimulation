#!/usr/bin/env python3
"""
Phase 4: Social - Context and Interaction Demonstration
Shows theory of mind, social learning, communication, and context sensitivity working together
"""

import numpy as np
import matplotlib.pyplot as plt
from theory_of_mind import TheoryOfMindManager
from social_learning import SocialLearningManager
from communication import CommunicationManager
from context_sensitivity import ContextSensitivityManager

def demonstrate_phase4():
    """Demonstrate Phase 4 capabilities"""
    print("\n" + "="*70)
    print("PHASE 4: SOCIAL - CONTEXT AND INTERACTION DEMONSTRATION")
    print("="*70)
    
    # Initialize managers
    tom_manager = TheoryOfMindManager()
    social_manager = SocialLearningManager()
    comm_manager = CommunicationManager()
    context_manager = ContextSensitivityManager()
    
    print("\n" + "-"*70)
    print("PART 1: THEORY OF MIND")
    print("-"*70)
    
    # Observe agents and infer mental states
    print("\nObserving Agent 1 behavior...")
    tom_manager.observe_agent(1, "cooperate", np.array([1.0, 1.0]))
    tom_manager.observe_agent(1, "cooperate", np.array([2.0, 1.0]))
    tom_manager.observe_agent(1, "share", np.array([2.0, 2.0]))
    
    print("Observing Agent 2 behavior...")
    tom_manager.observe_agent(2, "compete", np.array([3.0, 3.0]))
    tom_manager.observe_agent(2, "hoard", np.array([4.0, 3.0]))
    
    # Infer mental states
    tom_manager.infer_mental_state(1, observed_goals=["cooperation"])
    tom_manager.infer_mental_state(2, observed_goals=["competition"])
    
    mental_state1 = tom_manager.get_mental_state(1)
    mental_state2 = tom_manager.get_mental_state(2)
    
    print(f"\nAgent 1 Mental State:")
    print(f"   Beliefs: {len(mental_state1.beliefs) if mental_state1 else 0}")
    print(f"   Desires: {len(mental_state1.desires) if mental_state1 else 0}")
    
    print(f"\nAgent 2 Mental State:")
    print(f"   Beliefs: {len(mental_state2.beliefs) if mental_state2 else 0}")
    print(f"   Desires: {len(mental_state2.desires) if mental_state2 else 0}")
    
    # Recognize intentions
    tom_manager.intention_recognition.learn_intention_pattern("cooperate", ["cooperate", "cooperate", "share"])
    intentions1 = tom_manager.recognize_intention(1, ["cooperate", "cooperate", "share"])
    print(f"\nAgent 1 Intentions Recognized: {len(intentions1)}")
    
    print("\n" + "-"*70)
    print("PART 2: SOCIAL LEARNING")
    print("-"*70)
    
    # Observe demonstrations
    print("\nObserving demonstrations...")
    social_manager.observe_demonstration(1, ["cooperate", "share"], outcome=0.9)
    social_manager.observe_demonstration(2, ["compete", "hoard"], outcome=0.2)
    social_manager.observe_demonstration(1, ["cooperate", "help"], outcome=0.95)
    
    print(f"   Demonstrations observed: {len(social_manager.imitation_learning.demonstrations)}")
    
    # Receive feedback
    print("\nReceiving social feedback...")
    social_manager.receive_feedback(1, "cooperate", 0.9)
    social_manager.receive_feedback(1, "cooperate", 0.85)
    social_manager.receive_feedback(2, "compete", 0.3)
    
    cooperate_score = social_manager.social_reinforcement.get_action_score("cooperate")
    compete_score = social_manager.social_reinforcement.get_action_score("compete")
    print(f"   Cooperate score: {cooperate_score:.3f}")
    print(f"   Compete score: {compete_score:.3f}")
    
    # Create cultural practices
    print("\nCreating cultural practices...")
    social_manager.create_cultural_practice("greeting", "social", creator_id=1)
    social_manager.create_cultural_practice("sharing", "cooperation", creator_id=2)
    social_manager.create_cultural_practice("helping", "cooperation", creator_id=1)
    
    print(f"   Cultural practices: {len(social_manager.cultural_transmission.cultural_practices)}")
    
    # Transmit practices
    print("\nTransmitting practices...")
    social_manager.transmit_practice(1, 3, "greeting")
    social_manager.transmit_practice(2, 3, "sharing")
    print(f"   Transmissions: {len(social_manager.cultural_transmission.transmission_history)}")
    
    print("\n" + "-"*70)
    print("PART 3: COMMUNICATION")
    print("-"*70)
    
    # Create symbols
    print("\nCreating communication symbols...")
    comm_manager.create_symbol("help", "request_assistance")
    comm_manager.create_symbol("yes", "affirmation")
    comm_manager.create_symbol("cooperate", "collaboration")
    
    print(f"   Symbols created: {len(comm_manager.language_structures.symbols)}")
    
    # Send signals
    print("\nSending communication signals...")
    signal1 = comm_manager.send_signal(1, 'request', 'help', {'intent': 'request', 'urgency': 0.8}, receiver_id=2)
    signal2 = comm_manager.send_signal(2, 'inform', 'yes', {'intent': 'inform', 'certainty': 0.9}, receiver_id=1)
    signal3 = comm_manager.send_signal(1, 'inform', 'cooperate', {'intent': 'inform', 'certainty': 0.85}, receiver_id=2)
    
    print(f"   Signals sent: {len(comm_manager.signal_generation.generated_signals)}")
    
    # Receive and interpret
    print("\nReceiving and interpreting signals...")
    interp1, conf1 = comm_manager.receive_signal(signal1)
    interp2, conf2 = comm_manager.receive_signal(signal2)
    interp3, conf3 = comm_manager.receive_signal(signal3)
    
    print(f"   Signal 1: {interp1} (confidence: {conf1:.2f})")
    print(f"   Signal 2: {interp2} (confidence: {conf2:.2f})")
    print(f"   Signal 3: {interp3} (confidence: {conf3:.2f})")
    
    # Start conversation
    print("\nStarting conversation protocol...")
    comm_manager.communication_protocols.define_protocol("dialogue", [1, 2], ["request", "response"])
    conv_id = comm_manager.start_conversation("dialogue", [1, 2])
    print(f"   Conversation started: {conv_id >= 0}")
    
    print("\n" + "-"*70)
    print("PART 4: CONTEXT SENSITIVITY")
    print("-"*70)
    
    # Register contexts
    print("\nRegistering contexts...")
    context_manager.register_context("office", {"noise_level": 0.3, "light_level": 0.8, "people": 0.5})
    context_manager.register_context("home", {"noise_level": 0.1, "light_level": 0.6, "people": 0.2})
    context_manager.register_context("meeting", {"noise_level": 0.2, "light_level": 0.9, "people": 0.9})
    
    print(f"   Contexts registered: {len(context_manager.context_detection.known_contexts)}")
    
    # Add behavior rules
    print("\nAdding context-dependent behavior rules...")
    context_manager.add_behavior_rule("office", {"noise_level": 0.5}, "work_quietly", priority=1.0)
    context_manager.add_behavior_rule("office", {"light_level": 0.7}, "adjust_lighting", priority=0.8)
    context_manager.add_behavior_rule("meeting", {"people": 0.8}, "participate", priority=1.0)
    context_manager.add_behavior_rule("home", {"noise_level": 0.2}, "relax", priority=1.0)
    
    print(f"   Rules added: {sum(len(rules) for rules in context_manager.context_behavior.context_rules.values())}")
    
    # Detect context
    print("\nDetecting context from observed features...")
    observed = {"noise_level": 0.25, "light_level": 0.85, "people": 0.9}
    detected = context_manager.detect_context(observed)
    print(f"   Detected context: {detected}")
    
    # Select behavior
    print("\nSelecting behavior for context...")
    behavior = context_manager.select_behavior(detected or "office", observed)
    print(f"   Selected behavior: {behavior}")
    
    # Update situation
    print("\nUpdating situational awareness...")
    context_manager.update_situation("temperature", 22.0)
    context_manager.update_situation("humidity", 0.5)
    context_manager.update_situation("time_of_day", 0.7)
    
    summary = context_manager.situational_awareness.get_situational_summary()
    print(f"   Situational features tracked: {len(summary)}")
    
    # Switch contexts
    print("\nSwitching contexts...")
    context_manager.switch_context("office")
    context_manager.switch_context("meeting")
    print(f"   Context switches: {len(context_manager.context_switching.switch_history)}")
    
    print("\n" + "-"*70)
    print("INTEGRATED DEMONSTRATION")
    print("-"*70)
    
    # Show integrated capabilities
    print("\n📊 Integrated Statistics:")
    
    tom_stats = tom_manager.get_statistics()
    print(f"\nTheory of Mind:")
    print(f"   Agents tracked: {tom_stats['agents_tracked']}")
    print(f"   Intentions recognized: {tom_stats['intentions_recognized']}")
    print(f"   Perspectives modeled: {tom_stats['perspectives_modeled']}")
    
    social_stats = social_manager.get_statistics()
    print(f"\nSocial Learning:")
    print(f"   Demonstrations: {social_stats['demonstrations_observed']}")
    print(f"   Cultural practices: {social_stats['cultural_practices']}")
    print(f"   Teachers tracked: {social_stats['teachers_tracked']}")
    
    comm_stats = comm_manager.get_statistics()
    print(f"\nCommunication:")
    print(f"   Signals generated: {comm_stats['signals_generated']}")
    print(f"   Symbols: {comm_stats['symbols']}")
    print(f"   Active conversations: {comm_stats['active_conversations']}")
    
    context_stats = context_manager.get_statistics()
    print(f"\nContext Sensitivity:")
    print(f"   Known contexts: {context_stats['known_contexts']}")
    print(f"   Current context: {context_stats['current_context']}")
    print(f"   Context rules: {context_stats['context_rules']}")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('Phase 4: Social - Context and Interaction', fontsize=18, fontweight='bold')
    
    # Plot 1: Theory of Mind - Mental States
    ax1 = fig.add_subplot(gs[0, 0])
    if mental_state1 and mental_state2:
        agents = ['Agent 1', 'Agent 2']
        beliefs = [len(mental_state1.beliefs), len(mental_state2.beliefs)]
        desires = [len(mental_state1.desires), len(mental_state2.desires)]
        
        x = np.arange(len(agents))
        width = 0.35
        ax1.bar(x - width/2, beliefs, width, label='Beliefs', color='#3498DB', alpha=0.8)
        ax1.bar(x + width/2, desires, width, label='Desires', color='#2ECC71', alpha=0.8)
        ax1.set_title('Theory of Mind', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count')
        ax1.set_xticks(x)
        ax1.set_xticklabels(agents)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Social Learning - Behavior Scores
    ax2 = fig.add_subplot(gs[0, 1])
    behaviors = list(social_manager.social_reinforcement.behavior_scores.keys())
    scores = [social_manager.social_reinforcement.behavior_scores[b] for b in behaviors]
    if behaviors:
        ax2.bar(behaviors, scores, color='#2ECC71', alpha=0.8, edgecolor='black')
        ax2.set_title('Social Learning', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Score')
        ax2.set_xticklabels(behaviors, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Communication - Signal Types
    ax3 = fig.add_subplot(gs[0, 2])
    signals = comm_manager.signal_generation.generated_signals
    signal_types = {}
    for s in signals:
        signal_types[s.signal_type] = signal_types.get(s.signal_type, 0) + 1
    if signal_types:
        types = list(signal_types.keys())
        counts = [signal_types[t] for t in types]
        ax3.bar(types, counts, color='#3498DB', alpha=0.8, edgecolor='black')
        ax3.set_title('Communication', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Count')
        ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Context Sensitivity - Context Activations
    ax4 = fig.add_subplot(gs[1, 0])
    contexts = list(context_manager.context_detection.known_contexts.values())
    context_names = [c.name for c in contexts]
    activations = [c.activation for c in contexts]
    ax4.bar(context_names, activations, color='#F39C12', alpha=0.8, edgecolor='black')
    ax4.set_title('Context Detection', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Activation')
    ax4.set_xticklabels(context_names, rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.3)
    
    # Plot 5: Cultural Practices
    ax5 = fig.add_subplot(gs[1, 1])
    practices = list(social_manager.cultural_transmission.cultural_practices.values())
    practice_names = [p.rule for p in practices]
    frequencies = [p.frequency for p in practices]
    if practices:
        ax5.bar(range(len(practices)), frequencies, color='#E74C3C', alpha=0.8, edgecolor='black')
        ax5.set_title('Cultural Practices', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Frequency')
        ax5.set_xticks(range(len(practices)))
        ax5.set_xticklabels(practice_names, rotation=45, ha='right')
        ax5.grid(axis='y', alpha=0.3)
    
    # Plot 6: Communication Symbols
    ax6 = fig.add_subplot(gs[1, 2])
    symbols = list(comm_manager.language_structures.symbols.values())
    symbol_names = [s.symbol for s in symbols]
    frequencies = [s.frequency for s in symbols]
    if symbols:
        ax6.bar(range(len(symbols)), frequencies, color='#9B59B6', alpha=0.8, edgecolor='black')
        ax6.set_title('Communication Symbols', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Frequency')
        ax6.set_xticks(range(len(symbols)))
        ax6.set_xticklabels(symbol_names, rotation=45, ha='right')
        ax6.grid(axis='y', alpha=0.3)
    
    # Plot 7: Situational Awareness
    ax7 = fig.add_subplot(gs[2, 0])
    features = list(context_manager.situational_awareness.situational_features.keys())
    values = [context_manager.situational_awareness.situational_features[f] for f in features]
    if features:
        ax7.bar(range(len(features)), values, color='#1ABC9C', alpha=0.8, edgecolor='black')
        ax7.set_title('Situational Awareness', fontsize=12, fontweight='bold')
        ax7.set_ylabel('Value')
        ax7.set_xticks(range(len(features)))
        ax7.set_xticklabels(features, rotation=45, ha='right')
        ax7.grid(axis='y', alpha=0.3)
    
    # Plot 8: Demonstration Outcomes
    ax8 = fig.add_subplot(gs[2, 1])
    demos = list(social_manager.imitation_learning.demonstrations.values())
    outcomes = [d.outcome for d in demos]
    demo_ids = [d.demo_id for d in demos]
    if demos:
        colors = ['green' if o > 0.7 else 'red' if o < 0.4 else 'orange' for o in outcomes]
        ax8.bar(demo_ids, outcomes, color=colors, alpha=0.8, edgecolor='black')
        ax8.set_title('Demonstration Outcomes', fontsize=12, fontweight='bold')
        ax8.set_ylabel('Outcome')
        ax8.set_xlabel('Demo ID')
        ax8.grid(axis='y', alpha=0.3)
    
    # Plot 9: Summary Statistics
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    summary_text = "Phase 4 Summary:\n\n"
    summary_text += f"Theory of Mind:\n"
    summary_text += f"  Agents: {tom_stats['agents_tracked']}\n"
    summary_text += f"  Intentions: {tom_stats['intentions_recognized']}\n\n"
    summary_text += f"Social Learning:\n"
    summary_text += f"  Demos: {social_stats['demonstrations_observed']}\n"
    summary_text += f"  Practices: {social_stats['cultural_practices']}\n\n"
    summary_text += f"Communication:\n"
    summary_text += f"  Signals: {comm_stats['signals_generated']}\n"
    summary_text += f"  Symbols: {comm_stats['symbols']}\n\n"
    summary_text += f"Context:\n"
    summary_text += f"  Contexts: {context_stats['known_contexts']}\n"
    summary_text += f"  Rules: {context_stats['context_rules']}\n"
    
    ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax9.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.savefig('phase4_social_demonstration.png', dpi=300, bbox_inches='tight')
    print("\n✅ Visualization saved: phase4_social_demonstration.png")
    plt.close()
    
    print("\n" + "="*70)
    print("PHASE 4 DEMONSTRATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    demonstrate_phase4()

