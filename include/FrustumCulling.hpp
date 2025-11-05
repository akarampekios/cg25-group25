#pragma once

#include <glm/glm.hpp>
#include <array>

struct Plane {
    glm::vec3 normal;
    float distance;

    // Returns signed distance from point to plane
    float distanceToPoint(const glm::vec3& point) const {
        return glm::dot(normal, point) + distance;
    }
};

struct Frustum {
    std::array<Plane, 6> planes; // Left, Right, Bottom, Top, Near, Far

    // Extract frustum planes from view-projection matrix
    static Frustum fromViewProjection(const glm::mat4& viewProj) {
        Frustum frustum;
        
        // Left plane
        frustum.planes[0].normal = glm::vec3(viewProj[0][3] + viewProj[0][0],
                                             viewProj[1][3] + viewProj[1][0],
                                             viewProj[2][3] + viewProj[2][0]);
        frustum.planes[0].distance = viewProj[3][3] + viewProj[3][0];
        
        // Right plane
        frustum.planes[1].normal = glm::vec3(viewProj[0][3] - viewProj[0][0],
                                             viewProj[1][3] - viewProj[1][0],
                                             viewProj[2][3] - viewProj[2][0]);
        frustum.planes[1].distance = viewProj[3][3] - viewProj[3][0];
        
        // Bottom plane
        frustum.planes[2].normal = glm::vec3(viewProj[0][3] + viewProj[0][1],
                                             viewProj[1][3] + viewProj[1][1],
                                             viewProj[2][3] + viewProj[2][1]);
        frustum.planes[2].distance = viewProj[3][3] + viewProj[3][1];
        
        // Top plane
        frustum.planes[3].normal = glm::vec3(viewProj[0][3] - viewProj[0][1],
                                             viewProj[1][3] - viewProj[1][1],
                                             viewProj[2][3] - viewProj[2][1]);
        frustum.planes[3].distance = viewProj[3][3] - viewProj[3][1];
        
        // Near plane
        frustum.planes[4].normal = glm::vec3(viewProj[0][3] + viewProj[0][2],
                                             viewProj[1][3] + viewProj[1][2],
                                             viewProj[2][3] + viewProj[2][2]);
        frustum.planes[4].distance = viewProj[3][3] + viewProj[3][2];
        
        // Far plane
        frustum.planes[5].normal = glm::vec3(viewProj[0][3] - viewProj[0][2],
                                             viewProj[1][3] - viewProj[1][2],
                                             viewProj[2][3] - viewProj[2][2]);
        frustum.planes[5].distance = viewProj[3][3] - viewProj[3][2];
        
        // Normalize planes
        for (auto& plane : frustum.planes) {
            const float length = glm::length(plane.normal);
            plane.normal /= length;
            plane.distance /= length;
        }
        
        return frustum;
    }

    // Test if sphere is inside or intersecting frustum
    bool testSphere(const glm::vec3& center, float radius) const {
        for (const auto& plane : planes) {
            if (plane.distanceToPoint(center) < -radius) {
                return false; // Sphere is completely outside this plane
            }
        }
        return true; // Sphere is inside or intersecting frustum
    }

    // Test if AABB is inside or intersecting frustum
    bool testAABB(const glm::vec3& minBounds, const glm::vec3& maxBounds) const {
        for (const auto& plane : planes) {
            // Get the positive vertex (furthest in direction of plane normal)
            glm::vec3 positiveVertex;
            positiveVertex.x = plane.normal.x >= 0 ? maxBounds.x : minBounds.x;
            positiveVertex.y = plane.normal.y >= 0 ? maxBounds.y : minBounds.y;
            positiveVertex.z = plane.normal.z >= 0 ? maxBounds.z : minBounds.z;
            
            if (plane.distanceToPoint(positiveVertex) < 0) {
                return false; // AABB is completely outside this plane
            }
        }
        return true; // AABB is inside or intersecting frustum
    }
};

struct AABB {
    glm::vec3 min;
    glm::vec3 max;

    // Transform AABB by a matrix
    AABB transform(const glm::mat4& matrix) const {
        // Transform all 8 corners and find new min/max
        std::array<glm::vec3, 8> corners = {
            glm::vec3(min.x, min.y, min.z),
            glm::vec3(max.x, min.y, min.z),
            glm::vec3(min.x, max.y, min.z),
            glm::vec3(max.x, max.y, min.z),
            glm::vec3(min.x, min.y, max.z),
            glm::vec3(max.x, min.y, max.z),
            glm::vec3(min.x, max.y, max.z),
            glm::vec3(max.x, max.y, max.z),
        };

        glm::vec3 newMin = glm::vec3(matrix * glm::vec4(corners[0], 1.0f));
        glm::vec3 newMax = newMin;

        for (size_t i = 1; i < corners.size(); ++i) {
            glm::vec3 transformed = glm::vec3(matrix * glm::vec4(corners[i], 1.0f));
            newMin = glm::min(newMin, transformed);
            newMax = glm::max(newMax, transformed);
        }

        return {newMin, newMax};
    }

    // Get center of AABB
    glm::vec3 center() const {
        return (min + max) * 0.5f;
    }

    // Get radius of bounding sphere
    float radius() const {
        return glm::length(max - min) * 0.5f;
    }
};
