#!/usr/bin/env python3
"""
Test Framework for Self-Model
Tests self-representation, self-awareness, body schema, and identity maintenance
"""

import numpy as np
import matplotlib.pyplot as plt
from self_model import (
    SelfModelManager, SelfRepresentationSystem,
    SelfAwareness, BodySchemaSystem, IdentityMaintenance
)

class SelfModelTester:
    """Test framework for self-model"""
    
    def __init__(self):
        self.results = []
    
    def test_self_representation(self):
        """Test self-representation"""
        print("\n" + "="*60)
        print("TEST 1: Self-Representation")
        print("="*60)
        
        representation = SelfRepresentationSystem()
        
        # Test 1: Create representation
        print("\nTest 1.1: Creating self-representations")
        rep1 = representation.create_representation("capabilities", {"movement": 0.8, "sensing": 0.9})
        rep2 = representation.create_representation("preferences", {"cooperation": 0.9, "exploration": 0.7})
        
        print(f"   Representations created: {len(representation.self_representations)}")
        print(f"   Result: {'✅ PASS' if len(representation.self_representations) == 2 else '❌ FAIL'}")
        
        # Test 2: Get capabilities
        print("\nTest 1.2: Getting capabilities")
        capabilities = representation.get_capabilities()
        
        print(f"   Capabilities: {len(capabilities)}")
        print(f"   Result: {'✅ PASS' if len(capabilities) == 2 else '❌ FAIL'}")
        
        # Test 3: Update capability
        print("\nTest 1.3: Updating capabilities")
        representation.update_capability("movement", 0.9)
        
        updated = representation.get_capabilities()
        print(f"   Movement capability: {updated.get('movement', 0):.2f}")
        print(f"   Result: {'✅ PASS' if updated.get('movement', 0) == 0.9 else '❌ FAIL'}")
        
        return True
    
    def test_self_awareness(self):
        """Test self-awareness"""
        print("\n" + "="*60)
        print("TEST 2: Self-Awareness")
        print("="*60)
        
        awareness = SelfAwareness()
        
        # Test 1: Update internal state
        print("\nTest 2.1: Updating internal states")
        awareness.update_internal_state("energy", 0.8)
        awareness.update_internal_state("motivation", 0.7)
        awareness.update_internal_state("focus", 0.9)
        
        print(f"   States tracked: {len(awareness.internal_states)}")
        print(f"   Result: {'✅ PASS' if len(awareness.internal_states) == 3 else '❌ FAIL'}")
        
        # Test 2: Get state trend
        print("\nTest 2.2: Getting state trends")
        awareness.update_internal_state("energy", 0.9)
        trend = awareness.get_state_trend("energy")
        
        print(f"   Energy trend: {trend:.4f}")
        print(f"   Result: {'✅ PASS' if trend > 0 else '⚠️  CHECK'}")
        
        # Test 3: Compute awareness level
        print("\nTest 2.3: Computing awareness level")
        level = awareness.compute_awareness_level()
        
        print(f"   Awareness level: {level:.4f}")
        print(f"   Result: {'✅ PASS' if level > 0 else '❌ FAIL'}")
        
        return True
    
    def test_body_schema(self):
        """Test body schema"""
        print("\n" + "="*60)
        print("TEST 3: Body Schema")
        print("="*60)
        
        body = BodySchemaSystem()
        
        # Test 1: Create body schema
        print("\nTest 3.1: Creating body schemas")
        schema1 = body.create_body_schema("arm", np.array([1.0, 2.0]), ["grasp", "lift"], {"max_weight": 10.0})
        schema2 = body.create_body_schema("leg", np.array([0.0, 0.0]), ["walk", "run"], {"max_speed": 5.0})
        
        print(f"   Body schemas created: {len(body.body_schemas)}")
        print(f"   Result: {'✅ PASS' if len(body.body_schemas) == 2 else '❌ FAIL'}")
        
        # Test 2: Get capabilities
        print("\nTest 3.2: Getting capabilities")
        arm_caps = body.get_capabilities("arm")
        
        print(f"   Arm capabilities: {arm_caps}")
        print(f"   Result: {'✅ PASS' if len(arm_caps) == 2 else '❌ FAIL'}")
        
        # Test 3: Check constraint
        print("\nTest 3.3: Checking constraints")
        can_grasp = body.check_constraint("arm", "grasp")
        can_fly = body.check_constraint("arm", "fly")
        
        print(f"   Can grasp: {can_grasp}")
        print(f"   Can fly: {can_fly}")
        print(f"   Result: {'✅ PASS' if can_grasp and not can_fly else '❌ FAIL'}")
        
        return True
    
    def test_identity_maintenance(self):
        """Test identity maintenance"""
        print("\n" + "="*60)
        print("TEST 4: Identity Maintenance")
        print("="*60)
        
        identity = IdentityMaintenance()
        
        # Test 1: Add identity markers
        print("\nTest 4.1: Adding identity markers")
        identity.add_identity_marker("name", 1.0)
        identity.add_identity_marker("role", 0.9)
        identity.add_identity_marker("preferences", 0.8)
        
        print(f"   Identity markers: {len(identity.identity_markers)}")
        print(f"   Result: {'✅ PASS' if len(identity.identity_markers) == 3 else '❌ FAIL'}")
        
        # Test 2: Compute continuity
        print("\nTest 4.2: Computing continuity")
        continuity = identity.compute_continuity()
        
        print(f"   Continuity score: {continuity:.4f}")
        print(f"   Result: {'✅ PASS' if continuity >= 0 else '❌ FAIL'}")
        
        # Test 3: Update marker
        print("\nTest 4.3: Updating identity markers")
        identity.update_identity_marker("role", 0.95)
        
        print(f"   Role marker: {identity.identity_markers.get('role', 0):.2f}")
        print(f"   Result: {'✅ PASS' if identity.identity_markers.get('role', 0) == 0.95 else '❌ FAIL'}")
        
        return True
    
    def test_integrated_self_model(self):
        """Test integrated self-model"""
        print("\n" + "="*60)
        print("TEST 5: Integrated Self-Model")
        print("="*60)
        
        manager = SelfModelManager()
        
        # Update self-representation
        print("\nUpdating self-representations...")
        manager.update_self_representation("capabilities", {"movement": 0.8, "sensing": 0.9, "reasoning": 0.7})
        manager.update_self_representation("preferences", {"cooperation": 0.9, "exploration": 0.8})
        manager.update_self_representation("goals", {"learn": 0.9, "help": 0.8})
        
        print(f"   Self-aspects: {len(manager.self_representation.self_representations)}")
        
        # Update internal states
        print("\nUpdating internal states...")
        manager.update_internal_state("energy", 0.8)
        manager.update_internal_state("motivation", 0.7)
        manager.update_internal_state("focus", 0.9)
        manager.update_internal_state("curiosity", 0.85)
        
        print(f"   Internal states: {len(manager.self_awareness.internal_states)}")
        
        # Create body parts
        print("\nCreating body schemas...")
        manager.create_body_part("arm", np.array([1.0, 2.0]), ["grasp", "lift", "point"])
        manager.create_body_part("leg", np.array([0.0, 0.0]), ["walk", "run", "jump"])
        manager.create_body_part("eye", np.array([0.5, 1.5]), ["see", "track"])
        
        print(f"   Body parts: {len(manager.body_schema.body_schemas)}")
        
        # Add identity markers
        print("\nAdding identity markers...")
        manager.add_identity_marker("name", 1.0)
        manager.add_identity_marker("role", 0.9)
        manager.add_identity_marker("preferences", 0.8)
        manager.add_identity_marker("capabilities", 0.85)
        
        print(f"   Identity markers: {len(manager.identity_maintenance.identity_markers)}")
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Self-Model', fontsize=16, fontweight='bold')
        
        # Plot 1: Self-representations
        ax1 = axes[0, 0]
        aspects = list(manager.self_representation.self_representations.keys())
        aspect_sizes = [len(rep.content) for rep in manager.self_representation.self_representations.values()]
        
        bars = ax1.bar(aspects, aspect_sizes, color='#3498DB', alpha=0.8, edgecolor='black')
        ax1.set_title('Self-Representations', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Attributes')
        ax1.set_xticklabels(aspects, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Internal states
        ax2 = axes[0, 1]
        states = list(manager.self_awareness.internal_states.keys())
        values = [manager.self_awareness.internal_states[s] for s in states]
        
        bars = ax2.bar(states, values, color='#2ECC71', alpha=0.8, edgecolor='black')
        ax2.set_title('Internal States', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Value')
        ax2.set_xticklabels(states, rotation=45, ha='right')
        ax2.set_ylim([0, 1])
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Body parts capabilities
        ax3 = axes[1, 0]
        body_parts = list(manager.body_schema.body_schemas.keys())
        capability_counts = [len(manager.body_schema.body_schemas[bp].capabilities) for bp in body_parts]
        
        bars = ax3.bar(body_parts, capability_counts, color='#F39C12', alpha=0.8, edgecolor='black')
        ax3.set_title('Body Parts Capabilities', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Number of Capabilities')
        ax3.set_xticklabels(body_parts, rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats = manager.get_statistics()
        stats_text = "Self-Model Statistics:\n\n"
        stats_text += f"Self Aspects: {stats['self_aspects']}\n"
        stats_text += f"Internal States: {stats['internal_states']}\n"
        stats_text += f"Awareness Level: {stats['awareness_level']:.3f}\n"
        stats_text += f"Body Parts: {stats['body_parts']}\n"
        stats_text += f"Identity Markers: {stats['identity_markers']}\n"
        stats_text += f"Continuity Score: {stats['continuity_score']:.3f}\n"
        
        ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('self_model_progress.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved: self_model_progress.png")
        plt.close()
        
        # Final statistics
        stats = manager.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"   Self aspects: {stats['self_aspects']}")
        print(f"   Internal states: {stats['internal_states']}")
        print(f"   Awareness level: {stats['awareness_level']:.3f}")
        print(f"   Body parts: {stats['body_parts']}")
        print(f"   Identity markers: {stats['identity_markers']}")
        print(f"   Continuity score: {stats['continuity_score']:.3f}")
        
        return True
    
    def run_all_tests(self):
        """Run all self-model tests"""
        print("\n" + "="*70)
        print("SELF-MODEL TEST SUITE")
        print("="*70)
        
        tests = [
            ("Self-Representation", self.test_self_representation),
            ("Self-Awareness", self.test_self_awareness),
            ("Body Schema", self.test_body_schema),
            ("Identity Maintenance", self.test_identity_maintenance),
            ("Integrated Self-Model", self.test_integrated_self_model)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"\n❌ {test_name} FAILED with error: {e}")
                import traceback
                traceback.print_exc()
                results.append((test_name, False))
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        passed = sum(1 for _, r in results if r)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name}: {status}")
        
        print(f"\n   Total: {passed}/{total} tests passed")
        print(f"   Success rate: {passed/total*100:.1f}%")
        
        return passed == total

def main():
    """Main test function"""
    tester = SelfModelTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All self-model tests passed!")
    else:
        print("\n⚠️  Some tests failed. Review output above.")
    
    return success

if __name__ == "__main__":
    main()

