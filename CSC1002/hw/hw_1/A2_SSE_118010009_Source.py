# bokeh serve --show A2_SSE_118010009_Source.py
import string
import pymssql
from bokeh.core.properties import value
from bokeh.io import curdoc, show
from bokeh.layouts import layout, widgetbox as wb
from bokeh.models import ColumnDataSource, widgets as wd
from bokeh.models.widgets import Select
from bokeh.plotting import figure

colors = ['#c9d9d3', '#718dbf', '#e84d60']
gpa = ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D', 'F']
years = ['2015', '2016', '2017']
dct = dict(zip(gpa, range(9)))
STAdata = {
    'gpa': gpa,
    '2015': [0] * 9,
    '2016': [0] * 9,
    '2017': [0] * 9,
}

paragraph = wd.Paragraph(text='option')

def connectSQLServer():
    attr = dict(
        server='10.20.213.10',
        database='csc1002',
        user='csc1002',
        password='csc1002',
        port=1433,
        as_dict=True,
    )
    try:
        return pymssql.connect(**attr)
    except Exception as e:
        print(e)
        quit()


def fetchRows(cmd):
    sqlConn = connectSQLServer()
    with sqlConn.cursor(as_dict=True) as c:
        c.execute(cmd)
        rows = c.fetchall()
    return rows


def updateCourseData(rows):
    courseData = {}
    courseData['id'] = [r['course_id'] for r in rows]
    courseData['title'] = [r['title'] for r in rows]
    courseData['dept'] = [r['dept_name'] for r in rows]
    courseData['credits'] = [r['credits'] for r in rows]
    courseData['instructor'] = [
        r['instructor'] for r in rows
    ]
    table.source.data = courseData


# filter by button group
btnGroupLetters = wd.RadioButtonGroup(
    labels=list(string.ascii_uppercase), active=-1
)


def refreshByButton(new):
    letter = btnGroupLetters.labels[new]
    rows = fetchRows(
        "select * from lgu.course where title like '{}%'".format(
            letter
        )
    )
    updateCourseData(rows)


btnGroupLetters.on_click(refreshByButton)


# filter by text
filtr = ['begins with...', '...contains...', '...ends with']
btnGroupTitle = wd.RadioButtonGroup(labels=filtr, active=1)
btnGroupDept = wd.RadioButtonGroup(labels=filtr, active=1)
refresh = wd.Button(label='Refresh')
title_input = wd.TextInput(
    value='', title='Title:', placeholder=''
)
dept_input = wd.TextInput(
    value='', title='Department:', placeholder=''
)
optionGroup = wd.RadioGroup(
    labels=['and', 'or'], active=0, width=100, inline=True
)


def refreshByFilter():
    conj = optionGroup.labels[optionGroup.active]
    title, dept = title_input.value, dept_input.value

    titleCondition = "title like '{}'".format(
        [title + '%', '%' + title + '%', '%' + title][
            btnGroupTitle.active
        ]
    )
    deptCondition = "dept_name like '{}'".format(
        [dept + '%', '%' + dept + '%', '%' + dept][
            btnGroupTitle.active
        ]
    )
    rows = fetchRows(
        "select * from lgu.course where {} {} {}".format(
            titleCondition, conj, deptCondition
        )
    )
    updateCourseData(rows)


refresh.on_click(refreshByFilter)

# course info table
columns = [
    wd.TableColumn(field='id', title='Course ID'),
    wd.TableColumn(field='title', title='Title'),
    wd.TableColumn(field='dept', title='Department'),
    wd.TableColumn(field='credits', title='Credit'),
    wd.TableColumn(field='instructor', title='Instructor'),
]
table = wd.DataTable(
    source=ColumnDataSource(), columns=columns, width=800
)

# course info layout
courseInfo = layout(
    [
        [wb(btnGroupLetters, width=1000)],
        [wb(btnGroupTitle), wb(btnGroupDept)],
        [
            wb(title_input),
            wb(paragraph, optionGroup, width=100),
            wb(dept_input),
        ],
        [wb(refresh, width=100)],
        [wb(table)],
    ]
)

# statistics page widges
for stu in fetchRows(
    "select gpa, year from lgu.student where dept_name = 'accounting'"
    ):
    yr, g = stu['year'], dct[stu['gpa']]
    STAdata[yr][g] += 1
staSource = ColumnDataSource(STAdata)
rows = fetchRows('select * from lgu.course')
deptSelectList = sorted(
    list(
        set(  # fetch department name
            r['dept_name'] for r in rows
        )
    )
)

deptSelect = Select(
    title="Department:", value='', options=deptSelectList
)

p = figure(
    x_range=gpa,
    plot_height=500,
    title="GPA Counts by Year",
    toolbar_location=None,
    tools='',
)


p.vbar_stack(
    years,
    x='gpa',
    width=0.9,
    color=colors,
    source=staSource,
    legend=[value(y) for y in years],
)


def updateStatistics(attr, old, new):
    STAdata = {
        'gpa': gpa,
        '2015': [0] * 9,
        '2016': [0] * 9,
        '2017': [0] * 9,
    }
    rows = fetchRows(
        "select gpa, year from lgu.student where dept_name = '{}'".format(
            new
        )
    )
    for stu in rows:
        yr, g = stu['year'], dct[stu['gpa']]
        STAdata[yr][g] += 1
    staSource.data = STAdata


deptSelect.on_change('value', updateStatistics)


# statistics page layout
statistics = layout([[deptSelect, p]])

tab1 = wd.Panel(child=courseInfo, title='Course Info')
tab2 = wd.Panel(child=statistics, title='Statistics')
tabs = wd.Tabs(tabs=[tab1, tab2])

curdoc().add_root(tabs)
