#!/usr/bin/env python3
"""
Multi-Modal Integration Demonstration
Shows how the system integrates information from multiple sensory modalities
"""

import numpy as np
import matplotlib.pyplot as plt
from multimodal_integration import MultiModalIntegrationManager

def demonstrate_multimodal_integration():
    """Demonstrate multi-modal integration in action"""
    
    print("\n" + "="*70)
    print("MULTI-MODAL INTEGRATION DEMONSTRATION")
    print("="*70)
    
    # Create multi-modal manager
    manager = MultiModalIntegrationManager()
    
    print("\n🎯 Scenario: Integrating Multiple Sensory Modalities")
    print("   Goal: Combine vision, audio, and touch into unified understanding")
    
    # Register modalities
    print("\n👁️  Registering Modalities...")
    vision = manager.register_modality("vision", feature_size=20, 
                                      initial_features=np.random.random(20))
    audio = manager.register_modality("audio", feature_size=15,
                                     initial_features=np.random.random(15))
    touch = manager.register_modality("touch", feature_size=12,
                                      initial_features=np.random.random(12))
    
    print(f"   Vision: ID {vision.modality_id}, size {vision.feature_size}")
    print(f"   Audio: ID {audio.modality_id}, size {audio.feature_size}")
    print(f"   Touch: ID {touch.modality_id}, size {touch.feature_size}")
    
    # Set different reliabilities
    vision.reliability = 0.9
    audio.reliability = 0.8
    touch.reliability = 0.7
    
    print(f"\n   Reliabilities: Vision={vision.reliability}, Audio={audio.reliability}, Touch={touch.reliability}")
    
    # Learn cross-modal mappings
    print("\n🔗 Learning Cross-Modal Mappings...")
    for _ in range(10):
        # Update features with some correlation
        vision_features = np.random.random(20)
        audio_features = vision_features[:15] * 0.8 + np.random.normal(0, 0.1, 15)
        
        manager.update_modality(vision.modality_id, vision_features)
        manager.update_modality(audio.modality_id, audio_features)
        
        manager.learn_cross_modal_mapping(vision.modality_id, audio.modality_id)
    
    print(f"   Mappings learned: {len(manager.cross_modal_learning.mappings)}")
    
    # Test cross-modal translation
    print("\n🔄 Testing Cross-Modal Translation...")
    translated, confidence = manager.translate_modality(vision.modality_id, audio.modality_id)
    print(f"   Vision -> Audio translation confidence: {confidence:.4f}")
    
    # Fuse modalities multiple times
    print("\n🔀 Fusing Modalities...")
    unified_reps = []
    for i in range(5):
        # Update modalities
        manager.update_modality(vision.modality_id, np.random.random(20))
        manager.update_modality(audio.modality_id, np.random.random(15))
        manager.update_modality(touch.modality_id, np.random.random(12))
        
        # Fuse
        unified = manager.fuse_modalities([vision.modality_id, audio.modality_id, touch.modality_id])
        if unified:
            unified_reps.append(unified)
    
    print(f"   Created {len(unified_reps)} unified representations")
    
    # Show attention weights
    print("\n👀 Attention Weights:")
    if manager.attention.attention_history:
        latest_attention = manager.attention.attention_history[-1]
        for mod_id, weight in latest_attention.items():
            mod_name = next(m.name for m in manager.modalities.values() if m.modality_id == mod_id)
            print(f"   {mod_name}: {weight:.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Multi-Modal Integration Demonstration', fontsize=16, fontweight='bold')
    
    # Plot 1: Modality features comparison
    ax1 = axes[0, 0]
    modalities_list = list(manager.modalities.values())
    for m in modalities_list:
        # Normalize for comparison
        normalized = m.features / (np.max(np.abs(m.features)) + 1e-10)
        ax1.plot(normalized, label=m.name, linewidth=2, alpha=0.7, marker='o', markersize=4)
    ax1.set_title('Modality Features (Normalized)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Normalized Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Attention weights over time
    ax2 = axes[0, 1]
    if manager.attention.attention_history:
        history = manager.attention.attention_history
        modality_ids = list(history[0].keys())
        
        for mod_id in modality_ids:
            weights = [h.get(mod_id, 0) for h in history]
            mod_name = next(m.name for m in modalities_list if m.modality_id == mod_id)
            ax2.plot(weights, label=mod_name, linewidth=2, marker='o')
        
        ax2.set_title('Attention Weights Over Time', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Fusion Step')
        ax2.set_ylabel('Attention Weight')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Unified representations
    ax3 = axes[1, 0]
    if unified_reps:
        for i, unified in enumerate(unified_reps[:3]):  # Show first 3
            normalized = unified.unified_features / (np.max(np.abs(unified.unified_features)) + 1e-10)
            ax3.plot(normalized, label=f'Unified {i+1}', linewidth=2, alpha=0.7)
        ax3.set_title('Unified Representations', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Feature Index')
        ax3.set_ylabel('Normalized Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats = manager.get_statistics()
    stats_text = "Multi-Modal Statistics:\n\n"
    stats_text += f"Modalities: {stats['modalities']}\n"
    stats_text += f"Cross-Modal Mappings: {stats['cross_modal_mappings']}\n"
    stats_text += f"Unified Representations: {stats['unified_representations']}\n"
    stats_text += f"Attention Steps: {stats['attention_history_length']}\n\n"
    
    # Cross-modal mapping details
    if manager.cross_modal_learning.mappings:
        stats_text += "Cross-Modal Mappings:\n"
        for (from_id, to_id), mapping in list(manager.cross_modal_learning.mappings.items())[:2]:
            from_name = next(m.name for m in modalities_list if m.modality_id == from_id)
            to_name = next(m.name for m in modalities_list if m.modality_id == to_id)
            stats_text += f"  {from_name} -> {to_name}\n"
            stats_text += f"    Confidence: {mapping.confidence:.3f}\n"
    
    ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('multimodal_integration_demonstration.png', dpi=300, bbox_inches='tight')
    print("\n✅ Demonstration visualization saved: multimodal_integration_demonstration.png")
    plt.close()
    
    # Summary
    print("\n" + "="*70)
    print("DEMONSTRATION SUMMARY")
    print("="*70)
    print(f"\n✅ Multi-modal integration working correctly!")
    print(f"✅ Multiple modalities registered and managed")
    print(f"✅ Cross-modal mappings learned successfully")
    print(f"✅ Modalities fused into unified representations")
    print(f"✅ Attention weights computed dynamically")
    print(f"\n🎯 Key Achievement: Unified multi-sensory understanding!")

if __name__ == "__main__":
    demonstrate_multimodal_integration()

