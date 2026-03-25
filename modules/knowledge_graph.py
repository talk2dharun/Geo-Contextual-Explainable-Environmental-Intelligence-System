"""
GEEIS - Knowledge Graph Module
Parses WHO Guidelines PDF and builds a knowledge graph 
of pollutants, health impacts, and guidelines using NetworkX.
"""

import os
import re
import networkx as nx
from PyPDF2 import PdfReader
import json


# Pre-defined WHO guideline relationships
# These are extracted from standard WHO drinking water quality guidelines
WHO_KNOWLEDGE_BASE = {
    'pollutants': {
        'Arsenic': {
            'guideline_value': '0.01 mg/L',
            'health_impacts': ['Cancer', 'Skin lesions', 'Cardiovascular disease', 'Diabetes'],
            'category': 'Chemical'
        },
        'Nitrate': {
            'guideline_value': '50 mg/L',
            'health_impacts': ['Blue Baby Syndrome (Methemoglobinemia)', 'Thyroid effects'],
            'category': 'Chemical'
        },
        'Lead': {
            'guideline_value': '0.01 mg/L',
            'health_impacts': ['Neurological damage', 'Kidney damage', 'Developmental delays in children'],
            'category': 'Chemical'
        },
        'Mercury': {
            'guideline_value': '0.006 mg/L',
            'health_impacts': ['Neurological damage', 'Kidney damage', 'Reproductive effects'],
            'category': 'Chemical'
        },
        'Fluoride': {
            'guideline_value': '1.5 mg/L',
            'health_impacts': ['Dental fluorosis', 'Skeletal fluorosis'],
            'category': 'Chemical'
        },
        'Chlorine': {
            'guideline_value': '5 mg/L',
            'health_impacts': ['Taste and odour issues', 'Potential byproduct formation'],
            'category': 'Disinfectant'
        },
        'Chromium': {
            'guideline_value': '0.05 mg/L',
            'health_impacts': ['Cancer', 'Liver damage', 'Kidney damage'],
            'category': 'Chemical'
        },
        'Cadmium': {
            'guideline_value': '0.003 mg/L',
            'health_impacts': ['Kidney damage', 'Bone disease (Itai-itai)'],
            'category': 'Chemical'
        },
        'Copper': {
            'guideline_value': '2 mg/L',
            'health_impacts': ['Gastrointestinal effects', 'Liver damage'],
            'category': 'Chemical'
        },
        'E. coli': {
            'guideline_value': 'Must not be detectable in 100 mL',
            'health_impacts': ['Gastroenteritis', 'Hemolytic uremic syndrome'],
            'category': 'Microbiological'
        },
        'Total Coliform': {
            'guideline_value': 'Must not be detectable in 100 mL',
            'health_impacts': ['Gastrointestinal illness', 'Indicator of contamination'],
            'category': 'Microbiological'
        },
        'Trihalomethanes': {
            'guideline_value': 'Varies by compound',
            'health_impacts': ['Cancer risk', 'Liver effects', 'Kidney effects'],
            'category': 'Disinfection Byproduct'
        },
        'Turbidity': {
            'guideline_value': '<1 NTU (ideally <0.1 NTU)',
            'health_impacts': ['Pathogen shielding', 'Reduced disinfection effectiveness'],
            'category': 'Physical'
        },
        'pH': {
            'guideline_value': '6.5-8.5',
            'health_impacts': ['Corrosion of pipes', 'Metal leaching', 'Taste effects'],
            'category': 'Physical'
        },
        'Total Dissolved Solids': {
            'guideline_value': '<600 mg/L (good), <1000 mg/L (acceptable)',
            'health_impacts': ['Taste effects', 'Scaling', 'Laxative effects at high levels'],
            'category': 'Physical'
        },
        'Sulfate': {
            'guideline_value': '500 mg/L',
            'health_impacts': ['Taste effects', 'Laxative effects'],
            'category': 'Chemical'
        },
        'Pesticides': {
            'guideline_value': 'Varies by compound',
            'health_impacts': ['Cancer', 'Neurological effects', 'Reproductive effects'],
            'category': 'Chemical'
        },
        'Manganese': {
            'guideline_value': '0.4 mg/L',
            'health_impacts': ['Neurological effects', 'Taste and staining'],
            'category': 'Chemical'
        }
    },
    'water_quality_parameters': {
        'pH': {'safe_range': '6.5 - 8.5', 'measurement': 'pH units'},
        'Turbidity': {'safe_range': '<1 NTU', 'measurement': 'NTU'},
        'Conductivity': {'safe_range': '<400 uS/cm', 'measurement': 'uS/cm'},
        'TDS': {'safe_range': '<600 mg/L', 'measurement': 'mg/L'},
        'Chloramines': {'safe_range': '<4 mg/L', 'measurement': 'mg/L'},
        'Organic Carbon': {'safe_range': '<2 mg/L (treated)', 'measurement': 'mg/L'},
        'Hardness': {'safe_range': '<500 mg/L', 'measurement': 'mg/L CaCO3'}
    }
}


def parse_who_pdf(pdf_path):
    """
    Parse WHO guidelines PDF and extract key text content.
    Returns: list of text sections
    """
    if not os.path.exists(pdf_path):
        return []

    try:
        reader = PdfReader(pdf_path)
        sections = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and len(text.strip()) > 50:
                sections.append({
                    'page': i + 1,
                    'text': text.strip()
                })

        return sections

    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return []


def extract_pollutant_health_pairs(pdf_sections):
    """
    Extract pollutant-health impact pairs from parsed PDF text.
    Combines PDF extraction with pre-defined knowledge base.
    """
    pairs = []

    # Use pre-defined knowledge base
    for pollutant, info in WHO_KNOWLEDGE_BASE['pollutants'].items():
        for impact in info['health_impacts']:
            pairs.append({
                'pollutant': pollutant,
                'health_impact': impact,
                'guideline_value': info['guideline_value'],
                'category': info['category']
            })

    # Try to extract additional pairs from PDF
    if pdf_sections:
        pollutant_pattern = re.compile(
            r'(arsenic|nitrate|lead|mercury|fluoride|chlorine|chromium|'
            r'cadmium|copper|manganese|iron|zinc|selenium|barium|'
            r'antimony|nickel|uranium)',
            re.IGNORECASE
        )
        health_pattern = re.compile(
            r'(cancer|disease|damage|syndrome|disorder|effects?|toxicity|'
            r'illness|infection|contamination)',
            re.IGNORECASE
        )

        for section in pdf_sections:
            text = section['text']
            pollutants_found = pollutant_pattern.findall(text)
            health_terms = health_pattern.findall(text)

            if pollutants_found and health_terms:
                for pollutant in set(pollutants_found):
                    # Check if this pollutant is already in our knowledge base
                    if pollutant.title() not in WHO_KNOWLEDGE_BASE['pollutants']:
                        for health_term in set(health_terms):
                            pairs.append({
                                'pollutant': pollutant.title(),
                                'health_impact': f"Associated {health_term}",
                                'guideline_value': 'See WHO guidelines',
                                'category': 'Chemical',
                                'source': f'PDF Page {section["page"]}'
                            })

    return pairs


def build_knowledge_graph(pdf_path=None):
    """
    Build a NetworkX knowledge graph from WHO guidelines.
    Nodes: Pollutants, Health Impacts, Categories
    Edges: Relationships between them
    """
    G = nx.DiGraph()

    # Parse PDF if available
    pdf_sections = []
    if pdf_path:
        pdf_sections = parse_who_pdf(pdf_path)

    # Extract pairs
    pairs = extract_pollutant_health_pairs(pdf_sections)

    # Build graph
    for pair in pairs:
        pollutant = pair['pollutant']
        health_impact = pair['health_impact']
        category = pair['category']
        guideline = pair['guideline_value']

        # Add pollutant node
        G.add_node(pollutant, 
                   node_type='pollutant',
                   category=category,
                   guideline_value=guideline)

        # Add health impact node
        G.add_node(health_impact, 
                   node_type='health_impact')

        # Add category node
        G.add_node(category, 
                   node_type='category')

        # Add edges
        G.add_edge(pollutant, health_impact, 
                   relationship='causes',
                   guideline=guideline)

        G.add_edge(category, pollutant,
                   relationship='contains')

    # Add water quality parameter nodes
    for param, info in WHO_KNOWLEDGE_BASE['water_quality_parameters'].items():
        G.add_node(param,
                   node_type='parameter',
                   safe_range=info['safe_range'],
                   measurement=info['measurement'])

    return G


def get_health_risks(knowledge_graph, features_dict):
    """
    Given water quality features, identify potential health risks
    from the knowledge graph.
    """
    risks = []

    # Check pH
    ph = features_dict.get('ph', 7.0)
    if ph < 6.5 or ph > 8.5:
        if knowledge_graph.has_node('pH'):
            successors = list(knowledge_graph.successors('pH'))
            risks.append({
                'parameter': 'pH',
                'value': ph,
                'safe_range': '6.5 - 8.5',
                'status': 'Outside safe range',
                'potential_issues': successors[:3] if successors else ['Corrosion risk', 'Metal leaching']
            })

    # Check Turbidity
    turbidity = features_dict.get('Turbidity', 0)
    if turbidity > 5.0:
        risks.append({
            'parameter': 'Turbidity',
            'value': turbidity,
            'safe_range': '<5 NTU (WHO)',
            'status': 'Exceeds guideline',
            'potential_issues': ['Pathogen shielding', 'Reduced disinfection']
        })

    # Check TDS/Solids
    tds = features_dict.get('Solids', 0)
    if tds > 1000:
        risks.append({
            'parameter': 'Total Dissolved Solids',
            'value': tds,
            'safe_range': '<1000 mg/L',
            'status': 'Exceeds acceptable limit',
            'potential_issues': ['Taste effects', 'Potential health effects']
        })

    # Check Chloramines
    chloramines = features_dict.get('Chloramines', 0)
    if chloramines > 4.0:
        risks.append({
            'parameter': 'Chloramines',
            'value': chloramines,
            'safe_range': '<4 mg/L',
            'status': 'Exceeds safe limit',
            'potential_issues': ['Chemical exposure', 'Potential health effects']
        })

    # Check Conductivity
    conductivity = features_dict.get('Conductivity', 0)
    if conductivity > 400:
        risks.append({
            'parameter': 'Conductivity',
            'value': conductivity,
            'safe_range': '<400 uS/cm',
            'status': 'Above recommended',
            'potential_issues': ['High mineral content', 'Potential contamination']
        })

    # Check THMs
    thm = features_dict.get('Trihalomethanes', 0)
    if thm > 80:
        risks.append({
            'parameter': 'Trihalomethanes',
            'value': thm,
            'safe_range': '<80 ppb',
            'status': 'Exceeds limit',
            'potential_issues': ['Cancer risk', 'Liver effects']
        })

    # Check Organic Carbon
    oc = features_dict.get('Organic_carbon', 0)
    if oc > 4.0:
        risks.append({
            'parameter': 'Organic Carbon',
            'value': oc,
            'safe_range': '<4 mg/L (source water)',
            'status': 'Above source water limit',
            'potential_issues': ['Disinfection byproduct formation', 'Contamination indicator']
        })

    return risks


def get_graph_statistics(G):
    """Get basic statistics about the knowledge graph."""
    pollutants = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'pollutant']
    health_impacts = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'health_impact']
    categories = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'category']
    parameters = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'parameter']

    return {
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
        'pollutants': len(pollutants),
        'health_impacts': len(health_impacts),
        'categories': len(categories),
        'parameters': len(parameters),
        'pollutant_list': pollutants,
        'health_impact_list': health_impacts,
        'category_list': categories
    }
