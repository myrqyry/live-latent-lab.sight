#pragma once
#include "core/ocr_editor_geometry.hpp"
#include <QImage>
#include <QString>
#include <array>
#include <functional>
#include <optional>
class QWidget;
namespace sight {
struct EditorRegion {
 OcrEditorRect rect;
 QString name;
 bool enabled=false;
 bool operator==(const EditorRegion&) const=default;
};
using EditorRegions=std::array<EditorRegion,4>;
struct EditorPreview {QImage image;unsigned source_width=0,source_height=0;};
using EditorRequest=std::function<void()>;
using EditorRead=std::function<std::optional<EditorPreview>()>;
std::optional<EditorRegions> edit_ocr_regions(QWidget* parent,EditorRegions initial,
    unsigned source_width,unsigned source_height,EditorRequest request,EditorRead read);
}
