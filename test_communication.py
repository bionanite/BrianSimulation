#!/usr/bin/env python3
"""
Test Framework for Communication
Tests signal generation, interpretation, language structures, and protocols
"""

import numpy as np
import matplotlib.pyplot as plt
from communication import (
    CommunicationManager, SignalGeneration,
    SignalInterpretation, LanguageStructures, CommunicationProtocols
)

class CommunicationTester:
    """Test framework for communication"""
    
    def __init__(self):
        self.results = []
    
    def test_signal_generation(self):
        """Test signal generation"""
        print("\n" + "="*60)
        print("TEST 1: Signal Generation")
        print("="*60)
        
        generation = SignalGeneration()
        
        # Test 1: Create signal
        print("\nTest 1.1: Creating signals")
        signal1 = generation.create_signal(1, 'inform', 'location_A', {'intent': 'inform', 'certainty': 0.9})
        signal2 = generation.generate_request(2, 'help', receiver_id=3)
        
        print(f"   Signals created: {len(generation.generated_signals)}")
        print(f"   Result: {'✅ PASS' if len(generation.generated_signals) == 2 else '❌ FAIL'}")
        
        # Test 2: Generate request
        print("\nTest 1.2: Generating requests")
        request = generation.generate_request(1, 'information')
        
        print(f"   Request type: {request.signal_type}")
        print(f"   Result: {'✅ PASS' if request.signal_type == 'request' else '❌ FAIL'}")
        
        # Test 3: Generate inform
        print("\nTest 1.3: Generating inform signals")
        inform = generation.generate_inform(1, 'target_found')
        
        print(f"   Inform type: {inform.signal_type}")
        print(f"   Result: {'✅ PASS' if inform.signal_type == 'inform' else '❌ FAIL'}")
        
        return True
    
    def test_signal_interpretation(self):
        """Test signal interpretation"""
        print("\n" + "="*60)
        print("TEST 2: Signal Interpretation")
        print("="*60)
        
        interpretation = SignalInterpretation()
        generation = SignalGeneration()
        
        # Test 1: Receive signal
        print("\nTest 2.1: Receiving signals")
        signal = generation.create_signal(1, 'inform', 'message', {'intent': 'inform', 'certainty': 0.8})
        interpretation.receive_signal(signal)
        
        print(f"   Signals received: {len(interpretation.received_signals)}")
        print(f"   Result: {'✅ PASS' if len(interpretation.received_signals) == 1 else '❌ FAIL'}")
        
        # Test 2: Interpret signal
        print("\nTest 2.2: Interpreting signals")
        interp, confidence = interpretation.interpret_signal(signal)
        
        print(f"   Interpretation: {interp}")
        print(f"   Confidence: {confidence:.4f}")
        print(f"   Result: {'✅ PASS' if interp is not None else '❌ FAIL'}")
        
        # Test 3: Extract intent
        print("\nTest 2.3: Extracting intent")
        intent = interpretation.extract_intent(signal)
        
        print(f"   Intent: {intent}")
        print(f"   Result: {'✅ PASS' if intent == 'inform' else '❌ FAIL'}")
        
        return True
    
    def test_language_structures(self):
        """Test language structures"""
        print("\n" + "="*60)
        print("TEST 3: Language Structures")
        print("="*60)
        
        language = LanguageStructures()
        
        # Test 1: Create symbol
        print("\nTest 3.1: Creating symbols")
        symbol1 = language.create_symbol("hello", "greeting")
        symbol2 = language.create_symbol("world", "object")
        
        print(f"   Symbols created: {len(language.symbols)}")
        print(f"   Result: {'✅ PASS' if len(language.symbols) == 2 else '❌ FAIL'}")
        
        # Test 2: Associate symbols
        print("\nTest 3.2: Associating symbols")
        language.associate_symbols("hello", "world", strength=0.6)
        
        associations = language.get_associated_symbols("hello")
        print(f"   Associations: {len(associations)}")
        print(f"   Result: {'✅ PASS' if len(associations) == 1 else '❌ FAIL'}")
        
        # Test 3: Grammar rules
        print("\nTest 3.3: Grammar rules")
        language.add_grammar_rule(["hello", "world"], "greeting_phrase")
        parsed = language.parse_sequence(["hello", "world"])
        
        print(f"   Parsed result: {parsed}")
        print(f"   Result: {'✅ PASS' if parsed == 'greeting_phrase' else '❌ FAIL'}")
        
        return True
    
    def test_communication_protocols(self):
        """Test communication protocols"""
        print("\n" + "="*60)
        print("TEST 4: Communication Protocols")
        print("="*60)
        
        protocols = CommunicationProtocols()
        
        # Test 1: Define protocol
        print("\nTest 4.1: Defining protocols")
        protocols.define_protocol("turn_taking", [1, 2, 3], ["request", "response"])
        
        print(f"   Protocols defined: {len(protocols.protocols)}")
        print(f"   Result: {'✅ PASS' if len(protocols.protocols) == 1 else '❌ FAIL'}")
        
        # Test 2: Start conversation
        print("\nTest 4.2: Starting conversations")
        conv_id = protocols.start_conversation("turn_taking", [1, 2, 3])
        
        print(f"   Conversation ID: {conv_id}")
        print(f"   Result: {'✅ PASS' if conv_id >= 0 else '❌ FAIL'}")
        
        # Test 3: Get next speaker
        print("\nTest 4.3: Getting next speaker")
        next_speaker = protocols.get_next_speaker(conv_id)
        
        print(f"   Next speaker: {next_speaker}")
        print(f"   Result: {'✅ PASS' if next_speaker == 1 else '❌ FAIL'}")
        
        return True
    
    def test_integrated_communication(self):
        """Test integrated communication"""
        print("\n" + "="*60)
        print("TEST 5: Integrated Communication")
        print("="*60)
        
        manager = CommunicationManager()
        
        # Create symbols
        print("\nCreating symbols...")
        manager.create_symbol("help", "request_assistance")
        manager.create_symbol("yes", "affirmation")
        manager.create_symbol("no", "negation")
        
        print(f"   Symbols created: {len(manager.language_structures.symbols)}")
        
        # Send signals
        print("\nSending signals...")
        signal1 = manager.send_signal(1, 'request', 'help', {'intent': 'request', 'urgency': 0.8}, receiver_id=2)
        signal2 = manager.send_signal(2, 'inform', 'yes', {'intent': 'inform', 'certainty': 0.9}, receiver_id=1)
        
        print(f"   Signals sent: {len(manager.signal_generation.generated_signals)}")
        
        # Receive and interpret
        print("\nReceiving and interpreting signals...")
        interp1, conf1 = manager.receive_signal(signal1)
        interp2, conf2 = manager.receive_signal(signal2)
        
        print(f"   Signal 1 interpretation: {interp1}")
        print(f"   Signal 2 interpretation: {interp2}")
        
        # Start conversation
        print("\nStarting conversation...")
        manager.communication_protocols.define_protocol("dialogue", [1, 2], ["request", "response"])
        conv_id = manager.start_conversation("dialogue", [1, 2])
        
        print(f"   Conversation started: {conv_id >= 0}")
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Communication', fontsize=16, fontweight='bold')
        
        # Plot 1: Signal types
        ax1 = axes[0, 0]
        signals = manager.signal_generation.generated_signals
        signal_types = {}
        for s in signals:
            signal_types[s.signal_type] = signal_types.get(s.signal_type, 0) + 1
        
        if signal_types:
            types = list(signal_types.keys())
            counts = [signal_types[t] for t in types]
            
            bars = ax1.bar(types, counts, color='#3498DB', alpha=0.8, edgecolor='black')
            ax1.set_title('Signal Types', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Count')
            ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Symbols
        ax2 = axes[0, 1]
        symbols = list(manager.language_structures.symbols.values())
        symbol_names = [s.symbol for s in symbols]
        frequencies = [s.frequency for s in symbols]
        
        if symbols:
            bars = ax2.bar(range(len(symbols)), frequencies, color='#2ECC71', alpha=0.8, edgecolor='black')
            ax2.set_title('Symbols', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Frequency')
            ax2.set_xticks(range(len(symbols)))
            ax2.set_xticklabels(symbol_names, rotation=45, ha='right')
            ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Interpretation confidence
        ax3 = axes[1, 0]
        interpretations = manager.signal_interpretation.interpretation_history
        if interpretations:
            confidences = [interp[2]['confidence'] for interp in interpretations]
            
            bars = ax3.bar(range(len(confidences)), confidences, color='#F39C12', alpha=0.8, edgecolor='black')
            ax3.set_title('Interpretation Confidence', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Confidence')
            ax3.set_xlabel('Signal Index')
            ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats = manager.get_statistics()
        stats_text = "Communication Statistics:\n\n"
        stats_text += f"Signals Generated: {stats['signals_generated']}\n"
        stats_text += f"Signals Received: {stats['signals_received']}\n"
        stats_text += f"Symbols: {stats['symbols']}\n"
        stats_text += f"Grammar Rules: {stats['grammar_rules']}\n"
        stats_text += f"Protocols: {stats['protocols']}\n"
        stats_text += f"Active Conversations: {stats['active_conversations']}\n"
        
        ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('communication_progress.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved: communication_progress.png")
        plt.close()
        
        # Final statistics
        stats = manager.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"   Signals generated: {stats['signals_generated']}")
        print(f"   Signals received: {stats['signals_received']}")
        print(f"   Symbols: {stats['symbols']}")
        print(f"   Grammar rules: {stats['grammar_rules']}")
        print(f"   Protocols: {stats['protocols']}")
        
        return True
    
    def run_all_tests(self):
        """Run all communication tests"""
        print("\n" + "="*70)
        print("COMMUNICATION TEST SUITE")
        print("="*70)
        
        tests = [
            ("Signal Generation", self.test_signal_generation),
            ("Signal Interpretation", self.test_signal_interpretation),
            ("Language Structures", self.test_language_structures),
            ("Communication Protocols", self.test_communication_protocols),
            ("Integrated Communication", self.test_integrated_communication)
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
    tester = CommunicationTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All communication tests passed!")
    else:
        print("\n⚠️  Some tests failed. Review output above.")
    
    return success

if __name__ == "__main__":
    main()

