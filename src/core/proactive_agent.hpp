#pragma once

#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <optional>
#include "json.hpp"

// Use nlohmann::json for serialization
using json = nlohmann::json;

namespace lll {
namespace agent {

// ==========================================
// DATA STRUCTURES
// ==========================================

struct Attribute {
    std::string name;
    float importance_to_ask_score; // 0.0 to 1.0
    std::map<std::string, float> candidates; // value -> probability

    // JSON Serialization
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Attribute, name, importance_to_ask_score, candidates)
};

struct Entity {
    std::string name;
    std::string description;
    float importance_to_ask_score; // 0.0 to 1.0
    float probability_of_appearing; // 0.0 to 1.0
    std::string entity_type; // "explicit", "implicit", "background"
    
    // Attributes are parsed separately usually, but good to link them
    std::vector<Attribute> attributes;

    // JSON Serialization (excluding attributes for simple parsing)
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Entity, name, description, importance_to_ask_score, probability_of_appearing, entity_type)
};

struct Relation {
    std::string name; // "entity1-entity2"
    std::string description;
    float importance_to_ask_score;
    std::string name_entity_1;
    std::string name_entity_2;
    bool is_bidirectional;
    std::map<std::string, float> spatial_relation;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Relation, name, description, importance_to_ask_score, name_entity_1, name_entity_2, is_bidirectional, spatial_relation)
};

// ==========================================
// PROMPTS
// ==========================================

class ProactivePrompts {
public:
    static std::string get_entity_parser_prompt(const std::string& user_prompt) {
        // Adapted from google-deepmind/proactive_t2i_agents
        // Stripped examples for brevity in C++, kept core instructions
        std::stringstream ss;
        ss << "Given a text-to-image prompt list out all the entities that are mentioned in the prompt.\n\n"
           << "**Explicit Entities:** List all clearly stated entities (people, objects, animals, locations).\n"
           << "**Implicit Entities:** Identify potential entities that are implied.\n"
           << "**Background Entities:** Deduce relevant background elements (weather, location, time, mood).\n\n"
           << "Output a JSON list of objects with fields:\n"
           << "- name: Name of entity\n"
           << "- importance_to_ask_score: 0.0-1.0 (Higher if critical/uncertain)\n"
           << "- description: Short description\n"
           << "- entity_type: 'explicit', 'implicit', or 'background'\n"
           << "- probability_of_appearing: 0.0-1.0 (1.0 if definitely exists, 0.0 if explicitly excluded)\n\n"
           << "Input: { \"user_prompt\": \"" << user_prompt << "\" }\n"
           << "Output JSON:";
        return ss.str();
    }

    static std::string get_attribute_parser_prompt(const std::string& user_prompt, const std::string& entity_name, const std::string& existing_entities) {
        std::stringstream ss;
        ss << "Given a prompt and an entity, identify possible attributes.\n"
           << "1. Do not include existing entities as attributes.\n"
           << "2. Output a JSON list of attributes:\n"
           << "- name: Attribute name\n"
           << "- importance_to_ask_score: 0.0-1.0 (Higher if it influences other things or is ambiguous)\n"
           << "- candidates: Map of {value: probability}. generate at least 5 values if uncertain.\n\n"
           << "Input: {\n"
           << "  \"user_prompt\": \"" << user_prompt << "\",\n"
           << "  \"entity\": \"" << entity_name << "\",\n"
           << "  \"other_existing_entities\": \"" << existing_entities << "\"\n"
           << "}\n"
           << "Output JSON:";
        return ss.str();
    }
};

// ==========================================
// AGENT LOGIC
// ==========================================

class ProactiveAgent {
public:
    struct State {
        std::vector<Entity> entities;
        std::vector<Relation> relations;
        std::string refined_prompt;
    };

    State current_state;

    // TODO: Connect to actual LLM backend (Gemini/LocalGGUF)
    // This function parses the raw JSON response from the LLM
    std::vector<Entity> parse_entities(const std::string& json_response) {
        try {
            // Find start/end of JSON array in case LLM adds markdown
            size_t start = json_response.find('[');
            size_t end = json_response.rfind(']');
            if (start != std::string::npos && end != std::string::npos) {
                std::string raw = json_response.substr(start, end - start + 1);
                return json::parse(raw).get<std::vector<Entity>>();
            }
        } catch (...) {
            return {};
        }
        return {};
    }
};

} // namespace agent
} // namespace lll
