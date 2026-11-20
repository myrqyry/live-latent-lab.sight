#pragma once

#include <QApplication>
#include <QByteArray>
#include <QColor>
#include <QPalette>
#include <QString>
#include <QWidget>

namespace lll::ui {

struct ThemeColors {
    QColor window;
    QColor base;
    QColor alternateBase;
    QColor text;
    QColor mutedText;
    QColor button;
    QColor buttonText;
    QColor border;
    QColor accent;
    QColor accentText;
    QColor link;
};

inline ThemeColors themeColors(const QWidget *widget)
{
    const QPalette palette = widget ? widget->palette() : QApplication::palette();
    return {
        palette.color(QPalette::Window),
        palette.color(QPalette::Base),
        palette.color(QPalette::AlternateBase),
        palette.color(QPalette::Text),
        palette.color(QPalette::PlaceholderText),
        palette.color(QPalette::Button),
        palette.color(QPalette::ButtonText),
        palette.color(QPalette::Mid),
        palette.color(QPalette::Highlight),
        palette.color(QPalette::HighlightedText),
        palette.color(QPalette::Link),
    };
}

inline QColor iconColor(const QPalette &palette, bool onHighlight = false)
{
    if (onHighlight) {
        const QColor highlighted = palette.color(QPalette::HighlightedText);
        if (highlighted.isValid() && highlighted.lightness() >= 55)
            return highlighted;
    }

    for (const QPalette::ColorRole role : {QPalette::WindowText, QPalette::Text,
                                           QPalette::ButtonText}) {
        const QColor color = palette.color(role);
        if (color.isValid() && color.lightness() >= 55)
            return color;
    }

    return QColor(225, 225, 225);
}

inline QByteArray recolorSvg(const QByteArray &svg, const QColor &color)
{
    if (svg.isEmpty())
        return svg;

    const QByteArray value = color.name(QColor::HexArgb).toUtf8();
    QByteArray result = svg;
    result.replace("currentColor", value);
    result.replace("#292929", value);
    result.replace("stroke=\"#292929\"", "stroke=\"" + value + "\"");
    result.replace("fill=\"#292929\"", "fill=\"" + value + "\"");
    return result;
}

} // namespace lll::ui
